""" Utils for distributed training

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license

Thanks to Wonjae Kim for helping the distributed training implementation.
"""
import torch.distributed as dist
import torch


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor):
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank]


def grad_all_gather(tensor, cat=False):
    world_size = dist.get_world_size()
    gat = [torch.zeros_like(tensor) for _ in range(world_size)]
    AllGather.apply(gat, tensor)

    if cat:
        return torch.cat(gat, dim=0)
    else:
        return gat
