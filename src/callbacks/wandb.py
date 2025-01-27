from typing import Literal

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only


def _get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Get the W&B logger from the trainer.

    Raises:
        ValueError: If the W&B logger is not found.
    """
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            return logger

    raise ValueError("WandbLogger not found in spite of using wandb related callbacks.")


class ModelWatcher(Callback):
    """Watch the model with W&B at the start of training."""

    def __init__(
        self,
        *,
        log: Literal["gradients", "parameters", "all"] = "gradients",
        log_freq: int = 1000,
    ) -> None:
        self._log = log
        self._log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module) -> None:
        assert trainer.model is not None

        logger = _get_wandb_logger(trainer=trainer)
        logger.watch(
            model=trainer.model,
            log=self._log,
            log_freq=self._log_freq,
            log_graph=False,  # Setting this to False to avoid errors when used with ModelCheckpoint. cf. https://github.com/wandb/wandb/issues/2588
        )
