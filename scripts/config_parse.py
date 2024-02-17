"""Script to parse config file."""
import importlib.util
from types import ModuleType


def load_config_from_path(path: str) -> ModuleType:
    """Load config from path.

    Args:
    ----
        path (str): path to config file

    Returns:
    -------
        module: config module

    """
    spec = importlib.util.spec_from_file_location("Config", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def parse_config(config_path: str) -> dict:
    """Parse config file.

    Args:
    ----
        config_path: path to config file

    Returns:
    -------
        dict: parsed config

    """
    config = load_config_from_path(config_path)

    # If the config file has a base config, parse it and update the current config
    if hasattr(config, "_base_"):
        base_config = {}
        for base_config_path in config._base_:
            base_config.update(parse_config(base_config_path))

        base_config.update(config.CONFIG)
        return base_config
    else:
        return config.CONFIG


if __name__ == "__main__":
    config = parse_config("./config/darcy_flow.py")
    print(config)
