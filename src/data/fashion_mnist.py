from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningDataModule
from lightning.fabric.utilities.data import AttributeDict
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms


class _HParams(AttributeDict):
    data_dir: str
    train_val_ratio: tuple[float, float]
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    transform: Callable | None


class FashionMNISTDataModule(LightningDataModule):
    hparams: _HParams

    def __init__(
        self,
        *,
        data_dir: str = "data/",
        train_val_ratio: tuple[float, float] = (0.9, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()

        if sum(train_val_ratio) != 1.0:
            raise ValueError("Sum of train_val_ratio must be equal to 1.0.")

        self.save_hyperparameters(logger=False)

        self.transform = transforms.ToTensor() if transform is None else transform

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    def prepare_data(self) -> None:
        FashionMNIST(self.hparams.data_dir, train=True, download=True)
        FashionMNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val = random_split(
                dataset=FashionMNIST(self.hparams.data_dir, train=True, transform=self.transform),
                lengths=self.hparams.train_val_ratio,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_test = FashionMNIST(self.hparams.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader[Any]:
        assert self.data_train

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        assert self.data_val

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        assert self.data_test

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )
