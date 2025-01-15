from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from train import train


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
    train(cfg_train)
