import torch

def tensor_equal_eps(a:torch.Tensor,b:torch.Tensor,eps:float = 1e-9):
    return ((a-b)<eps).all
