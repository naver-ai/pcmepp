""" Triplet loss implemetation

Original code: https://github.com/woodfrog/vse_infty/blob/master/lib/loss.py
"""
import torch
import torch.nn as nn
from torch.autograd import Variable


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


class TripletLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, margin=0, max_violation=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s, **kwargs):
        if isinstance(im, dict):
            im = im['mean']
            s = s['mean']
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        _I = Variable(mask)
        if torch.cuda.is_available():
            _I = _I.cuda()
        cost_s = cost_s.masked_fill_(_I, 0)
        cost_im = cost_im.masked_fill_(_I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        loss = cost_s.sum() + cost_im.sum()
        loss_dict = {
            'loss': loss.data
        }
        return loss, loss_dict
