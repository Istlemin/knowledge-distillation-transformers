import torch
import math
from torch import nn

class TwnQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w : torch.Tensor, dim, clamp_val):
        if dim is None:
            dim = tuple(range(len(w.shape)))
        if type(dim) is int:
            dim = (dim,)
        ctx.save_for_backward(w, torch.tensor(clamp_val,device=w.device))
        n = math.prod((w.shape[d]) for d in dim)

        w = torch.clamp(w,-clamp_val,clamp_val)

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
        zeroing out gradient for clamped values.
        """
        w, clamp_val = ctx.saved_tensors
        grad_output *= ((-clamp_val < w) & (w < clamp_val))

        # Need to return one gradient for each argument,
        # but we only want one for [w] 
        return grad_output, None, None

class TwnQuantizer(nn.Module):
    def __init__(self, clamp_val=2.5):
        super().__init__()
        self.clamp_val = clamp_val

    def forward(self,w, dim=None):
        return TwnQuantizerFunction.apply(w,dim,self.clamp_val)


class MinMaxQuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w : torch.Tensor, dim, bits, clamp_val):
        if dim is None:
            dim = tuple(range(len(w.shape)))
        if type(dim) is int:
            dim = (dim,)
        ctx.save_for_backward(w, torch.tensor(clamp_val,device=w.device))

        w = torch.clamp(w,-clamp_val,clamp_val)
        
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
        zeroing out gradient for clamped values.
        """
        w,clamp_val = ctx.saved_tensors
        grad_output *= ((-clamp_val < w) & (w < clamp_val))

        # Need to return one gradient for each argument,
        # but we only want one for [w] 
        return grad_output, None, None, None

class MinMaxQuantizer(nn.Module):
    def __init__(self, bits=8, clamp_val=2.5):
        super().__init__()
        self.clamp_val = clamp_val
        self.bits=bits

    def forward(self,w, dim=None):
        return MinMaxQuantizerFunction.apply(w,dim,self.bits,self.clamp_val)


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
    def __init__(self, embedding : nn.Linear, quanter):
        super().__init__()
        self.weight = embedding.weight
        self.padding_idx=embedding.padding_idx
        self.quanter = quanter
    
    def forward(self, input):
        quant_weight = self.quanter(self.weight,-1)
        return nn.functional.embedding(input,quant_weight,padding_idx=self.padding_idx)



if __name__=="__main__":
    layer = QuantizedLinear(nn.Linear(64,64),weight_quanter=TwnQuantizer(),act_quanter=MinMaxQuantizer())
    torch.manual_seed(0)
    layer.weight = torch.nn.Parameter(torch.rand((64,64)))
    layer.bias = torch.nn.Parameter(torch.rand((64,)))
    inp = torch.rand((100,64))*10-5
    res = layer(inp)
    res = res**2
    res = res*10
    res = res+10
    res = res/res.sum()
    res = torch.exp(res)
    loss = res.sum() 
    print(loss.view(torch.int).sum())
    loss.backward()
    print(layer.weight.grad.view(torch.int).sum())