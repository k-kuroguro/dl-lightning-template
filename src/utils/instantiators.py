import hydra
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from .ranked_logger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_loggers(cfg: DictConfig) -> list[Logger]:
    """Instantiate loggers from the configuration.

    Args:
        cfg (DictConfig): Global configuration that may contain a "logger" key.

    Returns:
        list[Logger]: List of instantiated loggers.
    """
    if not cfg.get("logger"):
        log.info("No loggers found.")
        return []

    loggers = []
    for v in cfg.logger.values():
        log.info(f"Instantiating logger: {v._target_}")
        logger: Logger = hydra.utils.instantiate(v)
        loggers.append(logger)
    return loggers


def instantiate_callbacks(cfg: DictConfig) -> list[Callback]:
    """Instantiate callbacks from the configuration.

    Args:
        cfg (DictConfig): Global configuration that may contain a "callbacks" key.

    Returns:
        list[Callback]: List of instantiated callbacks.
    """
    if cfg.get("callbacks") is None:
        log.info("No callbacks found.")
        return []

    callbacks = []
    for v in cfg.callbacks.values():
        log.info(f"Instantiating callback: {v._target_}")
        callback: Callback = hydra.utils.instantiate(v)
        callbacks.append(callback)
    return callbacks
