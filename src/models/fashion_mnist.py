from typing import Callable, Final, TypedDict

import torch
from lightning import LightningModule
from lightning.fabric.utilities.data import AttributeDict
from lightning.pytorch.utilities.types import OptimizerConfig, OptimizerLRSchedulerConfig
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from .types import OptimizerFactory, SchedulerFactory


class _StepOutput(TypedDict):
    loss: torch.Tensor
    preds: torch.Tensor
    targets: torch.Tensor


class _HParams(AttributeDict):
    net: torch.nn.Module
    optimizer: OptimizerFactory
    scheduler: SchedulerFactory
    compile: bool


class FashionMNISTLitModule(LightningModule):
    NUM_CLASSES: Final = 10
    hparams: _HParams

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: OptimizerFactory,
        scheduler: SchedulerFactory,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net: torch.nn.Module | Callable = net

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.NUM_CLASSES)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None: ...

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> _StepOutput:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()  # type: ignore[func-returns-value]
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> _StepOutput:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self) -> None: ...

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> OptimizerConfig | OptimizerLRSchedulerConfig:
        assert self.trainer.model

        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
