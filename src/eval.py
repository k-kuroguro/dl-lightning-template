import warnings

import hydra
import lightning as L
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils import (
    RankedLogger,
    find_project_root,
    get_config_path,
    instantiate_callbacks,
    instantiate_loggers,
    register_custom_resolvers,
)

log = RankedLogger(__name__, rank_zero_only=True)

register_custom_resolvers()

rootutils.set_root(
    path=find_project_root(), project_root_env_var=True, dotenv=True, pythonpath=True
)


@hydra.main(version_base="1.3", config_path=str(get_config_path()), config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
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

    if cfg.get("logger") and cfg.logger.get("wandb"):
        import wandb

        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    main()
