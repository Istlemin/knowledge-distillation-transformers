import torch
import math
from torch import nn

def twn_quantizer(clamp_val=2.5):
    class TwnQuantizer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w : torch.Tensor, dim=None):
            if dim is None:
                dim = tuple(range(len(w.shape)))
            if type(dim) is int:
                dim = (dim,)
            ctx.save_for_backward(w)
            n = math.prod((w.shape[d]) for d in dim)

            w = torch.clamp(w,-clamp_val,clamp_val)

            thres = 0.7 * torch.norm(w, p=1, dim=dim) / n
            for d in dim:
                thres = thres.unsqueeze(d)

            b = (w>thres).type(w.dtype) - (w<-thres).type(w.dtype)
            alpha = torch.norm(b*w,p=1,dim=dim)/torch.norm(b,p=1,dim=dim)
            for d in dim:
                alpha = alpha.unsqueeze(d)

            return alpha*b

        @staticmethod
        def backward(ctx, grad_output):
            """
            Approximate the gradient wrt to the full-precision inputs
            using the gradient wrt to the quantized inputs, 
            zeroing out gradient for clamped values.
            """
            w = ctx.saved_tensors
            grad_output *= (-clamp_val < w < clamp_val)
            return grad_output, None
    
    return TwnQuantizer

def min_max_quantizer(bits=8, clamp_val=2.5):
    class MinMaxQuantizer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w : torch.Tensor, dim=None):
            if dim is None:
                dim = tuple(range(len(w.shape)))
            if type(dim) is int:
                dim = (dim,)
            ctx.save_for_backward(w)

            w = torch.clamp(w,-clamp_val,clamp_val)
            
            mn = mx = w
            for d in dim:
                mn = torch.min(mn,dim=d).values
                mx = torch.max(mx,dim=d).values
                mn = mn.unsqueeze(d)
                mx = mx.unsqueeze(d)

            round_factor = (2**bits-1)/(mx-mn)
            quant_w = torch.round((w-mn)*round_factor)/round_factor+mn

            return quant_w


        @staticmethod
        def backward(ctx, grad_output):
            """
            Approximate the gradient wrt to the full-precision inputs
            using the gradient wrt to the quantized inputs, 
            zeroing out gradient for clamped values.
            """
            w, = ctx.saved_tensors
            grad_output *= ((-clamp_val < w) & (w < clamp_val))
            return grad_output, None
            
    return MinMaxQuantizer

class QuantizedLinear(nn.Module):
    def __init__(self, linear : nn.Linear, weight_quanter, act_quanter=None):
        super().__init__()
        
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight_quanter = weight_quanter
        self.act_quanter = act_quanter

    def forward(self, input, input_quantize_dim=(-2,-1)):
        if self.act_quanter is not None:
            input = self.act_quanter.apply(input,input_quantize_dim)
        quant_weight = self.weight_quanter.apply(self.weight)
        return nn.functional.linear(input, quant_weight, self.bias)
    
class QuantizedEmbedding(nn.Module):
    def __init__(self, embedding : nn.Linear, quanter):
        super().__init__()
        self.weight = embedding.weight
        self.padding_idx=embedding.padding_idx
        self.quanter = quanter
    
    def forward(self, input):
        quant_weight = self.quanter.apply(self.weight,-1)
        return nn.functional.embedding(input,quant_weight,padding_idx=self.padding_idx)