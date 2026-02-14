import os
from typing import Any, Dict

from ..config import read_yaml
from ..config.json_handler import read_json
from ..config.toml_handler import load_toml


class Config:
    """Load all the configuration files of one type.
    Example:
    >>> config = Config(config_dir='config', extension='toml')
    >>> config.set('general.random', 0.5)
    >>> config.get('general.random')
    """
    def __init__(self, config_dir: str = None, extension: str = 'toml', single_config_path: str = None):
        """
        :param config_dir: directory of the configuration files
        :param extension: extension of the configuration files
        """
        self.config_dir = config_dir
        self.single_config_path = single_config_path
        self.extension = extension
        self.config_data: Dict[str, Any] = {}
        self.load_configs()

    def load_configs(self):
        """Load all the configuration files of one type."""
        # Find TOML files from self.config_dir
        if self.single_config_path:
            ext = os.path.splitext(self.single_config_path)[-1].lstrip('.').lower()
            if ext == 'toml':
                self._load_toml_file(self.single_config_path)
            elif ext == 'json':
                self._load_json_file(self.single_config_path)
            elif ext in ('yaml', 'yml'):
                self._load_yaml_file(self.single_config_path)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        else:
            config_files = [f for f in os.listdir(self.config_dir) if f.endswith(f'.{self.extension}')]
            # Read TOML files
            if self.extension == 'toml':
                for file_name in config_files:
                    file_path = os.path.join(self.config_dir, file_name)
                    if os.path.exists(file_path):
                        self._load_toml_file(file_path)
            # Read JSON files
            elif self.extension == 'json':
                for file_name in config_files:
                    file_path = os.path.join(self.config_dir, file_name)
                    if os.path.exists(file_path):
                        self._load_json_file(file_path)
            # Read YAML files
            elif self.extension == 'yaml':
                for file_name in config_files:
                    file_path = os.path.join(self.config_dir, file_name)
                    if os.path.exists(file_path):
                        self._load_yaml_file(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {self.extension}")

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

    def _load_toml_file(self, file_path: str):
        """Load a single TOML file and update the main config dictionary."""
        data = load_toml(file_path)
        self._merge_dicts(self.config_data, data)

    def _load_json_file(self, file_path: str):
        """Load a single JSON file and update the main config dictionary."""
        data = read_json(file_path)
        self._merge_dicts(self.config_data, data)

    def _load_yaml_file(self, file_path: str):
        """Load a single YAML file and update the main config dictionary."""
        data = read_yaml(file_path)
        self._merge_dicts(self.config_data, data)

    def _merge_dicts(self, main_dict: Dict[str, Any], new_data: Dict[str, Any]):
        """Recursively merge two dictionaries (new_data into main_dict)."""
        for key, value in new_data.items():
            if isinstance(value, dict) and key in main_dict:
                self._merge_dicts(main_dict[key], value)
            else:
                main_dict[key] = value
