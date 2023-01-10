import copy
import math
import torch
from torch import nn
from typing import Optional, Tuple
from transformers import BertConfig, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertSelfAttention

from modeling.quantization import (
    QuantizedEmbedding,
    QuantizedLinear,
    TwnQuantizer,
    MinMaxQuantizer,
)


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
            Q = self.act_quanter(Q, (1, 2, 3))
            K = self.act_quanter(K, (1, 2, 3))

        attention_scores = Q @ K.permute(
            (0, 1, 3, 2)
        )  # Shape is [batch_size, num_heads, seq_len, seq_len]
        if attention_mask is not None:
            attention_scores += attention_mask
        attention_probabilities = nn.functional.softmax(
            1 / math.sqrt(self.head_size) * attention_scores, dim=-1
        )
        attention_probabilities = self.dropout(attention_probabilities)

        if self.act_quanter is not None:
            attention_probabilities = self.act_quanter(
                attention_probabilities, (1, 2, 3)
            )
            V = self.act_quanter(V, (1, 2, 3))

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


def prepare_bert_for_kd(model: BertPreTrainedModel):
    for layer in model.bert.encoder.layer:
        layer.attention.self = CustomBertSelfAttention(
            layer.attention.self, model.config
        )
    return model


def prepare_bert_for_quantization(
    model: BertPreTrainedModel,
    weight_quanter=TwnQuantizer(),
    act_quanter=MinMaxQuantizer(bits=8),
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

    return model


def test_quantized_bert():
    """
    Assert that a 32-bit quantization gives same result as original model
    """
    from transformers import BertForMaskedLM
    from utils import set_random_seed

    config = BertConfig()

    model1 = BertForMaskedLM(config)
    model2 = copy.deepcopy(model1)
    model2 = prepare_bert_for_quantization(
        model2,
        weight_quanter=MinMaxQuantizer(bits=32, clamp_val=10000),
        act_quanter=MinMaxQuantizer(bits=32, clamp_val=10000),
    )

    inp = torch.randint(200, 1000, (10, 30))
    set_random_seed(0)
    out1 = model1(
        inp, return_dict=True, output_hidden_states=True, output_attentions=True
    )
    set_random_seed(0)
    out2 = model2(
        inp, return_dict=True, output_hidden_states=True, output_attentions=True
    )

    print(torch.abs(out1.logits - out2.logits).max())
    assert torch.abs(out1.logits - out2.logits).max() < 1e-5


def test_custom_attention():
    """
    Assert that kd model replicate behaviour of transformers library:
    """
    from transformers import BertForMaskedLM
    from utils import set_random_seed

    config = BertConfig()

    model1 = BertForMaskedLM(config)
    state_dict = model1.state_dict()
    model2 = prepare_bert_for_kd(BertForMaskedLM(config))
    model2.load_state_dict(state_dict)

    inp = torch.randint(200, 1000, (10, 30))
    set_random_seed(0)
    out1 = model1(inp)
    set_random_seed(0)
    out2 = model2(inp)

    assert torch.all(out1.logits - out2.logits < 1e-6)


if __name__ == "__main__":

    test_quantized_bert()
    test_custom_attention()

    print("Unit test passed")
