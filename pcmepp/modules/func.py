""" Orig code
https://github.com/woodfrog/vse_infty/blob/master/lib/encoders.py#L26-L31
"""
import torch


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
