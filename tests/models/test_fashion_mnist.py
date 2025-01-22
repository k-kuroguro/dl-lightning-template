import hydra
import torch
from omegaconf import DictConfig, open_dict
from torch import nn

from data.fashion_mnist import FashionMNISTDataModule
from models.fashion_mnist import FashionMNISTLitModule
from utils import find_project_root


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(in_features=28 * 28, out_features=10)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.fc(input)


def test_fashion_mnist_lit_module(cfg_train: DictConfig) -> None:
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True

    trainer = hydra.utils.instantiate(cfg_train.trainer)
    datamodule = FashionMNISTDataModule(data_dir=str(find_project_root() / "data"))

    model = FashionMNISTLitModule(
        net=Net(),
        optimizer=torch.optim.Adam,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        compile=False,
    )

    trainer.fit(model, datamodule=datamodule)
