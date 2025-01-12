from typing import Protocol

import torch
from torch.optim.optimizer import ParamsT as OptimizerParams


class OptimizerFactory(Protocol):
    def __call__(self, params: OptimizerParams) -> torch.optim.Optimizer: ...


class SchedulerFactory(Protocol):
    def __call__(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler: ...
