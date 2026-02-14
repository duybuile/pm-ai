# Read a yaml file

import yaml


def read_yaml(path: str) -> dict:
    """Read a yaml file"""
    with open(path, "r") as file:
        return yaml.safe_load(file)
