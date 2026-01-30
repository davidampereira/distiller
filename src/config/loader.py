"""Configuration loader for Finer project."""

import json
from pathlib import Path
from typing import Any

_config: dict[str, Any] | None = None


def load_config(config_path: str = "config.json") -> dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    global _config
    if _config is not None:
        return _config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        _config = json.load(f)

    return _config


def get(key: str, default: Any = None) -> Any:
    """Get a config value by dot-notation key.

    Args:
        key: Dot-separated key path (e.g., 'training.model').
        default: Default value if key not found.

    Returns:
        Configuration value or default.

    Examples:
        >>> get("training.model")
        'Qwen/Qwen3-4B'
        >>> get("training.lora.r")
        4
    """
    config = load_config()
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


def reload_config(config_path: str = "config.json") -> dict[str, Any]:
    """Force reload configuration from file.

    Useful for testing or when config file changes during runtime.
    """
    global _config
    _config = None
    return load_config(config_path)
