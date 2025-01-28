from typing import Literal

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from wandb.plot import confusion_matrix
from wandb.sdk.wandb_run import Run

from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


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


class MulticlassConfusionMatrixLogger(Callback):
    """Log the multiclass confusion matrix to W&B at the end of each test epoch."""

    def __init__(
        self,
        *,
        class_names: list[str] | None = None,
        title: str = "Confusion Matrix",
        split_table: bool = False,
    ) -> None:
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []
        self._class_names = class_names
        self._title = title
        self._split_table = split_table

    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        if outputs is None:
            raise ValueError('Expected "outputs" to be a dictionary with "preds" and "targets", but got None.')

        if isinstance(outputs, torch.Tensor):
            raise TypeError(f'Expected "outputs" to be a dictionary, but got {type(outputs)}.')

        try:
            preds = outputs["preds"]
            targets = outputs["targets"]
        except KeyError as e:
            raise KeyError('Expected "outputs" to contain "preds" and "targets" keys, but missing one or both.') from e

        if not isinstance(preds, torch.Tensor):
            raise TypeError(f'Expected "preds" to be of type torch.Tensor, but got {type(preds)}.')
        if not isinstance(targets, torch.Tensor):
            raise TypeError(f'Expected "targets" to be of type torch.Tensor, but got {type(targets)}.')

        self._preds.append(preds)
        self._targets.append(targets)

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module) -> None:
        all_preds = torch.cat(self._preds, dim=0)
        all_targets = torch.cat(self._targets, dim=0)

        logger = _get_wandb_logger(trainer=trainer)
        if isinstance(logger.experiment, Run):
            logger.experiment.log(
                {
                    "confusion_matrix": confusion_matrix(
                        preds=all_preds.tolist(),
                        y_true=all_targets.tolist(),
                        class_names=self._class_names,
                        title=self._title,
                        split_table=self._split_table,
                    ),
                },
            )
        else:
            log.warning("W&B is disabled. Multiclass confusion matrix will not be logged.")
