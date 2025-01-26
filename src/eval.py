import warnings
from typing import TYPE_CHECKING

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from constants import HYDRA_VERSION
from utils import RankedLogger, close_loggers, get_config_path, instantiate_callbacks, instantiate_loggers, pre_main

if TYPE_CHECKING:
    import lightning as L

log = RankedLogger(__name__, rank_zero_only=True)


def evaluate(cfg: DictConfig) -> dict[str, torch.Tensor] | None:
    if cfg.get("ignore_warnings"):
        log.info("Ignoring warnings ...")
        warnings.filterwarnings("ignore")

    if cfg.get("ckpt_path") is None:
        raise ValueError("ckpt_path is required for evaluation.")

    log.info(f"Instantiating model: {cfg.model._target_}")
    model: L.LightningModule = instantiate(cfg.model)

    log.info(f"Instantiating datamodule: {cfg.data._target_}")
    datamodule: L.LightningDataModule = instantiate(cfg.data)

    log.info("Instantiating loggers ...")
    loggers = instantiate_loggers(cfg)
    if loggers:
        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        for logger in loggers:
            logger.log_hyperparams(resolved_cfg)  # type: ignore[arg-type]

    log.info("Instantiating callbacks ...")
    callbacks = instantiate_callbacks(cfg)

    log.info(f"Instantiating trainer: {cfg.trainer._target_}")
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log.info("Starting testing ...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Testing is done.")

    close_loggers(loggers)

    return trainer.callback_metrics


@pre_main
@hydra.main(version_base=HYDRA_VERSION, config_path=str(get_config_path()), config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
