# This module defines some JSON handler functions
import json
import logging
import os

logger = logging.getLogger(__name__)


def read_json(json_file: str, encoding=None) -> dict:
    """
    Read a dictionary from a JSON file
    :param json_file:
    :return:
    """
    # Check if the file is a JSON file
    if not json_file.endswith('.json'):
        logger.error(f"Invalid file format. Expected a JSON file. {json_file}")
        raise ValueError("Invalid file format. Expected a JSON file.")
    try:
        with open(json_file, "r", encoding=encoding) as f:
            dic = json.load(f)
        return dic
    except Exception as e:
        logger.error(f"Error in reading from a JSON file {json_file}:{e}")
        raise e


def write_dict_to_json(dic: dict, json_file: str):
    """
    Write a dictionary to a JSON file
    :param dic:
    :param json_file:
    :return: None
    """
    try:
        # Check if the directory exists
        if not os.path.exists(os.path.dirname(json_file)):
            logger.debug(f"Creating directory: {os.path.dirname(json_file)}")
            os.makedirs(os.path.dirname(json_file))

        # Write data to a JSON file
        logger.debug(f"Writing dictionary to JSON file: {json_file}")
        with open(json_file, "w") as f:
            json.dump(dic, f)
    except Exception as e:
        logger.error(f"Error in writing to a JSON file {json_file}:{e}")
        raise e
