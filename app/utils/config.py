import json
import os
from typing import Any, Dict

# Default configuration
DEFAULT_CONFIG = {
    "embedding_model": "text-embedding-3-small",
    "model": "gpt-4o",
    "temperature": 0.7,
    "top_k": 3,
    "max_tokens": 1000,
}

# Path to configuration file
CONFIG_PATH = "data/config.json"


def load_config() -> Dict[str, Any]:
    """
    Load configuration from file or create default configuration.

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    # If config file doesn't exist, create it with default values
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    # Load config from file
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return DEFAULT_CONFIG


def save_config(config: Dict[str, Any]) -> None:
    """
    Save configuration to file.

    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    # Save config to file
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"Error saving configuration: {e}")
