"""
PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
from pcmepp.criterions.triplet import TripletLoss
from pcmepp.criterions.infonce import InfoNCELoss
from pcmepp.criterions.pcme import MCSoftContrastiveLoss
from pcmepp.criterions.pcmepp import ClosedFormSampledDistanceLoss


def get_criterion(
    name,
    margin=0.0, max_violation=False,
    init_shift=5, init_negative_scale=5,
    num_samples=8, vib_beta=0,
    prob_distance='energy_distance', pdist_fn='iterative',
    smoothness_alpha=0,
    **kwargs
):
    if name == 'triplet':
        return TripletLoss(margin=margin, max_violation=max_violation)
    elif name == 'info_nce':
        return InfoNCELoss(
            **kwargs)
    elif name == 'pcme':
        return MCSoftContrastiveLoss(
            init_shift=init_shift,
            init_negative_scale=init_negative_scale,
            num_samples=num_samples,
            vib_beta=vib_beta,
            pdist_fn=pdist_fn,
            prob_distance=prob_distance,
            **kwargs)
    elif name == 'pcmepp':
        return ClosedFormSampledDistanceLoss(
            init_shift=init_shift,
            init_negative_scale=init_negative_scale,
            num_samples=num_samples,
            vib_beta=vib_beta,
            smoothness_alpha=smoothness_alpha,
            prob_distance=prob_distance,
            **kwargs)
    else:
        raise ValueError(name)
