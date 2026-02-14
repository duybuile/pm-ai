# log_handler.py
# Configuration for console + file logging, with optional colored console output.

import logging
import os
from typing import Optional, List


LOG_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,  # prefer 'warning' over deprecated 'warn'
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

IGNORE_LOGGERS = [
    "requests.packages.urllib3.connectionpool",
    "urllib3.connectionpool",
    "git.cmd",
    "matplotlib.font_manager",
    "matplotlib.pyplot",
    "PIL.PngImagePlugin",
    "werkzeug",
]


class ColoredFormatter(logging.Formatter):
    """
    Adds ANSI color codes to levelname and message based on severity.
    Only used for console logging when enabled.
    Ensures original record is restored so other handlers (e.g., file) are unaffected.
    """
    COLOR_MAP = {
        logging.DEBUG: "\033[94m",  # Blue
        logging.INFO: "\033[92m",  # Green
        logging.WARNING: "\033[93m",  # Yellow
        logging.ERROR: "\033[91m",  # Red
        logging.CRITICAL: "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLOR_MAP.get(record.levelno, self.RESET)

        # Preserve originals (raw template + args)
        orig_levelname = record.levelname
        orig_msg_raw = record.msg
        orig_args = record.args

        # Render the original message once (without color)
        rendered = record.getMessage()

        try:
            # Colorize levelname and the *rendered* message
            record.levelname = f"{color}{orig_levelname}{self.RESET}"
            record.msg = f"{color}{rendered}{self.RESET}"
            record.args = ()  # since msg is already rendered
            return super().format(record)
        finally:
            # Restore for other handlers (e.g., file)
            record.levelname = orig_levelname
            record.msg = orig_msg_raw
            record.args = orig_args


def _coerce_level(level: str) -> int:
    return LOG_LEVEL.get(str(level).lower(), logging.INFO)


def _apply_ignores(ignore: List[str]):
    for name in ignore:
        logging.getLogger(name).setLevel(logging.WARNING)


def _make_formatter(colored: bool) -> logging.Formatter:
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    return ColoredFormatter(fmt) if colored else logging.Formatter(fmt)


def setup_logger(
        *,
        console_logging: bool = True,
        filestream_logging: bool = False,
        filepath: str = "",
        level: str = "info",
        mode: str = "a",
        colored_console: bool = True,
        logger_name: Optional[str] = None,
        reset_handlers: bool = True,
        ignore: List = None,
) -> logging.Logger:
    """
    Configure logging for console and/or file.

    Parameters
    ----------
    console_logging : bool
        Enable logging to console.
    filestream_logging : bool
        Enable logging to a file.
    filepath : str
        Path to the log file (required if filestream_logging=True).
    level : str
        One of {'debug','info','warning','error','critical'}.
    mode : str
        File open mode ('a' append, 'w' overwrite).
    colored_console : bool
        Apply ANSI colors to console output.
    logger_name : Optional[str]
        Name of the logger to configure. None → root logger.
    reset_handlers : bool
        If True, remove existing handlers on the target logger before adding new ones.
    ignore : List
        List of loggers to ignore (i.e. to set to INFO)

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    if filestream_logging and not filepath:
        raise ValueError("`filepath` is required when filestream_logging=True")

    logger = logging.getLogger(logger_name)  # root if None
    logger.setLevel(_coerce_level(level))
    if ignore is None:
        ignore = []
    ignore_list = list(set(ignore + IGNORE_LOGGERS))
    _apply_ignores(ignore_list)

    if reset_handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # Prevent double messages if configuring a named logger
    # (child loggers will still propagate to root unless you configure them too)
    logger.propagate = logger_name is None

    if console_logging:
        ch = logging.StreamHandler()
        ch.setLevel(_coerce_level(level))
        ch.setFormatter(_make_formatter(colored_console))
        logger.addHandler(ch)

    if filestream_logging:
        # Ensure directory exists
        log_dir = os.path.dirname(os.path.abspath(filepath))
        if log_dir and not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(filepath, mode=mode, encoding="utf-8")
        fh.setLevel(_coerce_level(level))
        fh.setFormatter(_make_formatter(False))  # no color in files
        logger.addHandler(fh)

    return logger


# Backwards-compatible helpers (optional)
def setup_console_logger(level: str = "info", colored: bool = True) -> logging.Logger:
    return setup_logger(console_logging=True, filestream_logging=False, level=level, colored_console=colored)


def setup_filestream_logger(file: str, level: str = "info", mode: str = "a") -> logging.Logger:
    return setup_logger(console_logging=False, filestream_logging=True, filepath=file, level=level, mode=mode,
                        colored_console=False)


def setup_both_logger(file: str, level: str = "info", mode: str = "a") -> logging.Logger:
    return setup_logger(console_logging=True, filestream_logging=True, filepath=file, level=level, mode=mode,
                        colored_console=True)
