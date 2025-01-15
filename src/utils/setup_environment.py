from typing import Callable


def _get_setup_environment() -> Callable[[], None]:
    initialized = False

    def setup_environment() -> None:
        nonlocal initialized
        if initialized:
            return None

        import rootutils

        from utils import (
            find_project_root,
            register_custom_resolvers,
        )

        register_custom_resolvers()

        rootutils.set_root(
            path=find_project_root(), project_root_env_var=True, dotenv=True, pythonpath=True
        )

        initialized = True

    return setup_environment


setup_environment = _get_setup_environment()
