from pathlib import Path
from typing import Generator

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from constants import HYDRA_VERSION
from utils import find_project_root, get_config_path


def _override_cfg(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.paths.root_dir = str(find_project_root())

        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 0.01
        cfg.trainer.limit_val_batches = 0.1
        cfg.trainer.limit_test_batches = 0.1
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1

        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False

        cfg.logger = None
        cfg.experiment = None
        cfg.sweeper = None

        cfg.ckpt_path = None
        cfg.seed = None


def _override_train_cfg(cfg: DictConfig) -> None:
    _override_cfg(cfg)

    with open_dict(cfg):
        cfg.train = True
        cfg.eval = False


def _override_eval_cfg(cfg: DictConfig) -> None:
    _override_cfg(cfg)

    with open_dict(cfg):
        ...


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    with initialize_config_dir(
        version_base=HYDRA_VERSION,
        config_dir=str(get_config_path()),
    ):
        cfg = compose(config_name="train.yaml", return_hydra_config=True, overrides=[])
        _override_train_cfg(cfg)

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    with initialize_config_dir(
        version_base=HYDRA_VERSION,
        config_dir=str(get_config_path()),
    ):
        cfg = compose(config_name="eval.yaml", return_hydra_config=True, overrides=[])
        _override_eval_cfg(cfg)

    return cfg


@pytest.fixture(scope="function")
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> Generator[DictConfig, None, None]:
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> Generator[DictConfig, None, None]:
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
