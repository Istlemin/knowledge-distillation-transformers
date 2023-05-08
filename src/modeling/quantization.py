import math
import torch
from torch import nn
from typing import Optional, Tuple

from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertSelfAttention

def clip_and_save(ctx, w, clip_val):
    ctx.save_for_backward((w<-clip_val) | (w>clip_val))
    return torch.clamp(w,-clip_val,clip_val)

def gradient_apply_clipping(ctx, grad_output):
    clip_mask, = ctx.saved_tensors
    return grad_output * clip_mask

class TwnQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w : torch.Tensor, dim, clip_val):
        w = clip_and_save(ctx, w, clip_val)

        if dim is None:
            dim = tuple(range(len(w.shape)))
        if type(dim) is int:
            dim = (dim,)
        n = math.prod((w.shape[d]) for d in dim)


        thres = torch.norm(w, p=1, dim=dim) / n * 0.7
        for d in dim:
            thres = thres.unsqueeze(d)

        b = (w>thres).type(w.dtype) - (w<-thres).type(w.dtype)
        alpha = torch.sum(torch.abs(b*w),dim=dim)/torch.sum(torch.abs(b),dim=dim)
        for d in dim:
            alpha = alpha.unsqueeze(d)

        return alpha*b

    @staticmethod
    def backward(ctx, grad_output):
        """
        Approximate the gradient wrt to the full-precision inputs
        using the gradient wrt to the quantized inputs, 
        zeroing out gradient for clipped values.
        """
        # Need to return one gradient for each argument,
        # but we only want one for [w] 
        return gradient_apply_clipping(ctx, grad_output), None, None

class TwnQuantizer(nn.Module):
    def __init__(self, clip_val=2.5):
        super().__init__()
        self.clip_val = clip_val

    def forward(self,w, dim=None):
        return TwnQuantizerFunction.apply(w,dim,self.clip_val)


class MinMaxQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w : torch.Tensor, dim, bits, clip_val):
        w = clip_and_save(ctx, w, clip_val)

        if dim is None:
            dim = tuple(range(len(w.shape)))
        if type(dim) is int:
            dim = (dim,)
        
        mn = mx = w
        for d in dim:
            mn = torch.min(mn,dim=d).values
            mx = torch.max(mx,dim=d).values
            mn = mn.unsqueeze(d)
            mx = mx.unsqueeze(d)

        alpha = (mx-mn + 1e-8)
        size = (2**bits-1)
        quant_w = torch.round((w-mn)/alpha*size)/size*alpha+mn

        return quant_w


    @staticmethod
    def backward(ctx, grad_output):
        """
        Approximate the gradient wrt to the full-precision inputs
        using the gradient wrt to the quantized inputs, 
        zeroing out gradient for clipped values.
        """
        # Need to return one gradient for each argument,
        # but we only want one for [w] 
        return gradient_apply_clipping(ctx, grad_output), None, None, None

class MinMaxQuantizer(nn.Module):
    def __init__(self, bits=8, clip_val=2.5):
        super().__init__()
        self.clip_val = clip_val
        self.bits=bits

    def forward(self,w, dim=None):
        return MinMaxQuantizerFunction.apply(w,dim,self.bits,self.clip_val)


class QuantizedLinear(nn.Module):
    def __init__(self, linear : nn.Linear, weight_quanter, act_quanter=None):
        super().__init__()
        
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight_quanter = weight_quanter
        self.act_quanter = act_quanter

    def forward(self, input, input_quantize_dim=None):
        if self.act_quanter is not None:
            input = self.act_quanter(input,input_quantize_dim)
        quant_weight = self.weight_quanter(self.weight)
        return nn.functional.linear(input, quant_weight, self.bias)
    
class QuantizedEmbedding(nn.Module):
    def __init__(self, embedding : nn.Embedding, quanter):
        super().__init__()
        self.weight = embedding.weight
        self.padding_idx=embedding.padding_idx
        self.quanter = quanter
    
    def forward(self, input):
        quant_weight = self.quanter(self.weight,-1)
        return nn.functional.embedding(input,quant_weight,padding_idx=self.padding_idx)


class CustomBertSelfAttention(nn.Module):
    """
    Custom rewritten self-attention module compatible with the transformers library,
    with following changes:
        - Returns attention scores (pre softmax) instead of attention probabilities (post softmax)
        - Supports quantization from TernaryBERT
    """

    def __init__(
        self,
        old_self_attention: BertSelfAttention,
        config: BertConfig,
        weight_quanter=None,
        act_quanter=None,
    ):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("Hidden size not divisible by number of attention heads")

        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // self.num_heads

        self.weight_quanter = weight_quanter
        self.act_quanter = act_quanter

        self.query = old_self_attention.query
        self.key = old_self_attention.key
        self.value = old_self_attention.value
        if self.weight_quanter is not None:
            self.query = QuantizedLinear(self.query, self.weight_quanter, act_quanter)
            self.key = QuantizedLinear(self.key, self.weight_quanter, act_quanter)
            self.value = QuantizedLinear(self.value, self.weight_quanter, act_quanter)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        assert head_mask is None
        assert encoder_hidden_states is None
        assert encoder_attention_mask is None
        assert past_key_value is None

        (batch_size, seq_len, hidden_size) = hidden_states.shape

        QKV_shape = (batch_size, seq_len, self.num_heads, self.head_size)
        Q = self.query(hidden_states).view(QKV_shape).permute((0, 2, 1, 3))
        K = self.key(hidden_states).view(QKV_shape).permute((0, 2, 1, 3))
        V = self.value(hidden_states).view(QKV_shape).permute((0, 2, 1, 3))
        # Q,K,V are now shape [batch_size, self.num_heads, seq_len, head_size]

        if self.act_quanter is not None:
            Q = self.act_quanter(Q, None)#(1, 2, 3))
            K = self.act_quanter(K, None)#(1, 2, 3))

        attention_scores = Q @ K.permute(
            (0, 1, 3, 2)
        )  # Shape is [batch_size, num_heads, seq_len, seq_len]
        attention_scores /= math.sqrt(self.head_size)
        if attention_mask is not None:
            attention_scores += attention_mask
        attention_probabilities = nn.functional.softmax(attention_scores, dim=-1)
        attention_probabilities = self.dropout(attention_probabilities)

        if self.act_quanter is not None:
            attention_probabilities = self.act_quanter(
                attention_probabilities, None#(1, 2, 3)
            )
            V = self.act_quanter(V, None)#(1, 2, 3))

        result = (
            attention_probabilities @ V
        )  # Shape is [batch_size, num_heads, seq_len, head_size]
        result = result.permute((0, 2, 1, 3)).reshape(
            (batch_size, seq_len, hidden_size)
        )  # Shape is [batch_size, seq_len, hidden_size]

        if output_attentions:
            return (result, attention_scores)
        else:
            return (result,)

def prepare_bert_for_quantization(
    model: BertPreTrainedModel,
    weight_quanter=TwnQuantizer(clip_val=2.5),
    act_quanter=MinMaxQuantizer(bits=8, clip_val=2.5),
):
    config = model.config
    
    model.bert.embeddings.word_embeddings = QuantizedEmbedding(
        model.bert.embeddings.word_embeddings, weight_quanter
    )
    for layer in model.bert.encoder.layer:
        layer.attention.self = CustomBertSelfAttention(
            layer.attention.self,
            config,
            weight_quanter=weight_quanter,
            act_quanter=act_quanter,
        )
        layer.attention.output.dense = QuantizedLinear(
            layer.attention.output.dense,
            weight_quanter=weight_quanter,
            act_quanter=act_quanter,
        )
        layer.intermediate.dense = QuantizedLinear(
            layer.intermediate.dense,
            weight_quanter=weight_quanter,
            act_quanter=act_quanter,
        )
        layer.output.dense = QuantizedLinear(
            layer.output.dense, weight_quanter=weight_quanter, act_quanter=act_quanter
        )
    model.bert.pooler.dense = QuantizedLinear(
        model.bert.pooler.dense, weight_quanter=weight_quanter, act_quanter=act_quanter
    )
    return model