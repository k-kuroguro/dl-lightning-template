import os
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from .paths import find_project_root

if TYPE_CHECKING:
    from collections.abc import Callable


def register_custom_resolvers() -> None:
    resolvers: tuple[tuple[str, Callable, dict], ...] = (
        ("find_project_root", lambda: str(find_project_root()), {"use_cache": True}),
        ("basename", lambda x: os.path.basename(str(x).rstrip("/")), {}),
    )
    for name, func, kwargs in resolvers:
        OmegaConf.register_new_resolver(name, func, **kwargs)
