import functools
from collections.abc import Callable
from typing import Any

from lightning.pytorch.loggers import Logger, WandbLogger


def pre_main(main_func: Callable) -> Callable:
    """A decorator to perform common initialization tasks before executing the main function.

    This executes:
        - Register OmegaConf custom resolvers.
        - Set the project root path and add it to pythonpath.
        - Read .env file and set environment variables.
    """

    @functools.wraps(main_func)
    def wrapper(*args, **kwargs) -> Any:
        import rootutils

        from .hydra_resolvers import register_custom_resolvers
        from .paths import find_project_root

        register_custom_resolvers()
        rootutils.set_root(path=find_project_root(), project_root_env_var=False, dotenv=True, pythonpath=True)

        return main_func(*args, **kwargs)

    return wrapper


def close_loggers(loggers: list[Logger]) -> None:
    """Explicitly closes any active loggers.

    Args:
        loggers (list[Logger]): List of active loggers.
    """
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            import wandb

            if wandb.run:
                wandb.finish()
