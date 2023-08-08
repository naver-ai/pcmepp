""" Improved probabilistic embedding loss for cross-modal retrieval

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
import warnings

import torch
import torch.nn as nn


class ClosedFormSampledDistanceLoss(nn.Module):
    def __init__(
            self,
            init_shift=5,
            init_negative_scale=5,
            vib_beta=0,
            smoothness_alpha=0,
            prob_distance='csd',
            **kwargs):
        super().__init__()

        shift = init_shift * torch.ones(1)
        negative_scale = init_negative_scale * torch.ones(1)

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.vib_beta = vib_beta
        self.smoothness_alpha = smoothness_alpha

        # XXX Do not specify prob_distance unless for the prob dist ablation study
        self.prob_distance = prob_distance

        self.bceloss = nn.BCEWithLogitsLoss()

        if self.prob_distance not in {'csd', 'wdist'}:
            raise ValueError(f'Invalid prob_distance. Expected ("csd", "wdist"), but {prob_distance=}')

    def max_violation_on(self):
        warnings.warn(
            'PCME loss does not support max violation. Nothing happens')
        return

    def max_violation_off(self):
        warnings.warn(
            'PCME loss does not support max violation. Nothing happens')
        return

    def kl_divergence(self, mu, logsigma):
        kl_loss = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).mean()
        if kl_loss > 10000:
            # XXX prevent loss exploration
            warnings.warn(f'Detected a VIB loss explosion ({kl_loss=} > 10000). Ignore the VIB loss for stability.')
            return 0
        return kl_loss

    def _recompute_matched(self, matched, logits, smoothness=0):
        """ Recompute the `matched` matrix if the smoothness value is given.
        """
        if not smoothness:
            return matched, None
        else:
            logits = logits.view(matched.size())
            # XXX Warning: all negative pairs will return weird results
            gt_labels, gt_indices = torch.max(matched, dim=1)
            gt_vals = logits[:, gt_indices].diag()
            pseudo_gt_indices = (logits >= gt_vals.unsqueeze(1))
            new_matched = (gt_labels.unsqueeze(1) * (pseudo_gt_indices))
            _matched = matched.clone()
            _matched[pseudo_gt_indices] = new_matched[pseudo_gt_indices]

            return _matched, torch.sum(pseudo_gt_indices).item() - len(gt_indices)

    def _compute_prob_matching_loss(self, logits, matched, smoothness=0):
        matched, n_pseudo_gts = self._recompute_matched(matched, logits, smoothness)
        loss = self.bceloss(logits, matched)

        return {
            'loss': loss,
            'n_pseudo_gts': n_pseudo_gts,
        }

    def _compute_closed_form_loss(self, input1, input2, matched, smoothness=0):
        """ Closed-form probabilistic matching loss -- See Eq (1) and (2) in the paper.
        """
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1)
        sigma_pdist = ((torch.exp(input1['std']).unsqueeze(1) + torch.exp(input2['std']).unsqueeze(0))).sum(-1)
        logits = mu_pdist + sigma_pdist
        logits = -self.negative_scale * logits + self.shift
        loss_dict = self._compute_prob_matching_loss(logits, matched, smoothness=smoothness)
        loss_dict['loss/mu_pdist'] = mu_pdist.mean()
        loss_dict['loss/sigma_pdist'] = sigma_pdist.mean()
        return loss_dict

    def _compute_wd_loss(self, input1, input2, matched, smoothness=0):
        """ Wasserstien loss (only used for the ablation study)
        """
        mu_pdist = ((input1['mean'].unsqueeze(1) - input2['mean'].unsqueeze(0)) ** 2).sum(-1).view(-1)
        sigma_pdist = ((torch.exp(input1['std'] / 2).unsqueeze(1) - torch.exp(input2['std'] / 2).unsqueeze(0)) ** 2).sum(-1).view(-1)

        logits = mu_pdist + sigma_pdist
        logits = logits.reshape(len(input1['mean']), len(input2['mean']))
        logits = -self.negative_scale * logits + self.shift
        loss_dict = self._compute_prob_matching_loss(logits, matched, smoothness=smoothness)
        loss_dict['loss/mu_pdist'] = mu_pdist.mean()
        loss_dict['loss/sigma_pdist'] = sigma_pdist.mean()
        return loss_dict

    def forward(self, img_emb, cap_emb, matched=None):
        if self.prob_distance == 'wdist':
            loss_fn = self._compute_wd_loss
        else:
            loss_fn = self._compute_closed_form_loss
        vib_loss = 0

        if self.vib_beta != 0:
            vib_loss =\
                self.kl_divergence(img_emb['mean'], img_emb['std']) + \
                self.kl_divergence(cap_emb['mean'], cap_emb['std'])

        if matched is None:
            matched = torch.eye(len(img_emb['mean'])).to(img_emb['mean'].device)

        loss = loss_fn(img_emb, cap_emb, matched=matched)
        # NOTE: Efficient implementation for
        # when i2t loss and t2i loss are the same (https://github.com/naver-ai/pcme/issues/3)
        loss = 2 * loss['loss'] + self.vib_beta * vib_loss

        loss_dict = {
            'loss/loss': loss,
            'criterion/shift': self.shift,
            'criterion/negative_scale': self.negative_scale,
        }

        if self.vib_beta != 0:
            loss_dict['loss/vib_loss'] = vib_loss

        if self.smoothness_alpha:
            smooth_i2t_loss = loss_fn(img_emb, cap_emb, matched=matched, smoothness=self.smoothness_alpha)
            smooth_t2i_loss = loss_fn(cap_emb, img_emb, matched=matched.T, smoothness=self.smoothness_alpha)
            loss = loss + self.smoothness_alpha * (smooth_i2t_loss['loss'] + smooth_t2i_loss['loss'])
            loss_dict['loss/loss'] = loss
            loss_dict['loss/n_pseudo_gts'] = smooth_i2t_loss['n_pseudo_gts'] + smooth_t2i_loss['n_pseudo_gts']

        return loss, loss_dict
