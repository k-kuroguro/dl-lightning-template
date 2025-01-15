import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from eval import evaluate
from train import train


@pytest.mark.slow
def test_train_and_eval(cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2

    HydraConfig().set_config(cfg_train)
    train_metrics = train(cfg_train)

    assert train_metrics is not None
    assert "test/acc" in train_metrics
    assert "last.ckpt" in os.listdir(os.path.join(cfg_train.paths.log_dir, "checkpoints"))

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = os.path.join(cfg_train.paths.log_dir, "checkpoints", "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metrics = evaluate(cfg_eval)

    assert test_metrics is not None
    assert "test/acc" in test_metrics
    assert test_metrics["test/acc"] > 0.0
    assert train_metrics["test/acc"] == pytest.approx(test_metrics["test/acc"], abs=1e-3)
