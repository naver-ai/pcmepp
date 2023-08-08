""" InfoNCE loss implementation

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(
            self,
            init_tau=1,
            **kwargs):
        super().__init__()

        tau = init_tau * torch.ones(1)
        tau = nn.Parameter(tau)

        self.register_parameter('tau', tau)

    def max_violation_on(self):
        warnings.warn(
            'InfoNCE loss does not support max violation. Nothing happens')
        return

    def max_violation_off(self):
        warnings.warn(
            'InfoNCE loss does not support max violation. Nothing happens')
        return

    def forward(self, img_emb, cap_emb, distributed=False, **kwargs):
        img_emb = img_emb['mean']
        cap_emb = cap_emb['mean']
        cos_sim = img_emb @ cap_emb.T
        logits = cos_sim * self.tau
        gts = torch.arange(len(logits), dtype=torch.long, device=logits.device)
        if distributed:
            loss = F.cross_entropy(logits, gts)
        else:
            loss = (F.cross_entropy(logits, gts) + F.cross_entropy(logits.T, gts)) / 2

        loss_dict = {
            'loss/loss': loss,
            'criterion/tau': self.tau,
        }

        return loss, loss_dict
