from pathlib import Path

import pytest
import torch

from data.fashion_mnist import FashionMNISTDataModule
from utils import find_project_root


@pytest.mark.parametrize(["batch_size", "train_val_ratio"], [[32, (0.9, 0.1)], [128, (0.7, 0.3)]])
def test_fashion_mnist_datamodule(batch_size: int, train_val_ratio: tuple[float, float]) -> None:
    TRAIN_SET_NUM = 60_000
    TEST_SET_NUM = 10_000
    OVERALL_SET_NUM = TRAIN_SET_NUM + TEST_SET_NUM

    data_dir = find_project_root() / "data"

    dm = FashionMNISTDataModule(
        data_dir=str(data_dir), batch_size=batch_size, train_val_ratio=train_val_ratio
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "FashionMNIST").exists()
    assert Path(data_dir, "FashionMNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    assert len(dm.data_train) == pytest.approx(TRAIN_SET_NUM * train_val_ratio[0])
    assert len(dm.data_val) == pytest.approx(TRAIN_SET_NUM * train_val_ratio[1])
    assert len(dm.data_test) == TEST_SET_NUM

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == OVERALL_SET_NUM

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
