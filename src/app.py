"""Top-level Streamlit entrypoint.

Run with:
    streamlit run app.py
"""
from src import cfg
from src.ui.st_app import run_ui
from utils.config.log_handler import setup_logger

logger = setup_logger(
    level=cfg.get("logging.level", "info"),
    console_logging=cfg.get("logging.console_logging", True)
)

if __name__ == "__main__":
    logger.info("Starting app...")
    run_ui()
