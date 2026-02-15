import logging
import os

logger = logging.getLogger(__name__)


def check_and_make_directory(directory_path: str) -> str:
    """
    Check if the path exists and if not, create it
    :param directory_path: the path to check and create
    :return: the path
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def check_and_make_file(file_path: str) -> str:
    """
    Check if the parent directory of a file_path exists and if not, create the directory of the file path
    :param file_path: the file_path to check
    :return: the directory path of the file_path
    """
    path = os.path.dirname(file_path)
    return check_and_make_directory(path)


def list_files(directory_path: str) -> list:
    """
    List all files in a directory
    :param directory_path: the directory to list
    :return: a list of files (with the full path)
    """
    # Check if directory_path exists
    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist")
        raise ValueError(f"Directory {directory_path} does not exist")
    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def list_files_per_extension(directory_path: str, extension: str) -> list:
    """
    List all files in a directory per extension
    :param directory_path: the directory to list
    :param extension: the extension to list
    :return: a list of files (with the full path)
    """
    if extension == "":
        logger.warning("Empty extension provided, returning empty list")
        return []
    files = list_files(directory_path)
    return [f for f in files if f.endswith(extension)]
