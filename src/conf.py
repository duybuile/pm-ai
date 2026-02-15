# Read config file
import glob
import logging
import os
from typing import Any, Dict

from utils.config.toml_handler import load_toml

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.config_data: Dict[str, Any] = {}
        self.load_configs()

    def load_configs(self, config_files: list[str] = None):
        """Load all the TOML configuration files."""
        # Find all *toml files in the config directory
        if config_files is None:
            config_files = glob.glob(os.path.join(self.config_dir, "*.toml"))

        for file_name in config_files:
            file_path = (
                file_name
                if os.path.isabs(file_name) or os.path.dirname(file_name)
                else os.path.join(self.config_dir, file_name)
            )
            if os.path.exists(file_path):
                self._load_toml_file(file_path)

    def _load_toml_file(self, file_path: str):
        """Load a single TOML file and update the main config dictionary."""
        try:
            data = load_toml(file_path)
            self._merge_dicts(self.config_data, data)
        except Exception as e:
            logger.exception("Error loading config file %s", file_path)
            raise e

    def _merge_dicts(self, main_dict: Dict[str, Any], new_data: Dict[str, Any]):
        """Recursively merge two dictionaries (new_data into main_dict)."""
        for key, value in new_data.items():
            if isinstance(value, dict) and key in main_dict:
                self._merge_dicts(main_dict[key], value)
            else:
                main_dict[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the configuration using dot notation."""
        keys = key.split('.')
        data = self.config_data
        for k in keys:
            if k in data:
                data = data[k]
            else:
                return default
        return data

    def set(self, key: str, value: Any):
        """Set a value in the configuration using dot notation."""
        keys = key.split('.')
        data = self.config_data
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value

    def get_all(self) -> Dict[str, Any]:
        """Return the entire configuration dictionary."""
        return self.config_data
