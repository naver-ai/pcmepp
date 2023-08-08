"""Logger wrapper for PyTorch experiments.

PCME++
Copyright (c) 2023-present NAVER Cloud Corp.
MIT license
"""
from datetime import datetime

from typing import Dict, Optional

import torch

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


class PCMEPPLogger(TensorBoardLogger):
    def __init__(self, save_dir, **kwargs):
        super().__init__(save_dir=save_dir, **kwargs)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.experiment.add_scalars(k, v, step)
            else:
                try:
                    self.experiment.add_scalar(k, v, step)
                except Exception as ex:
                    m = f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    raise ValueError(m) from ex

        print(f"{datetime.now()} {step=} {metrics=}")
