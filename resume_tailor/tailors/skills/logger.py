import logging
from typing import Optional

def get_logger(name: Optional[str] = None, log_file: Optional[str] = None) -> logging.Logger:
    """Creates and configures a logger instance.

    Args:
        name: Optional name for the logger.
        log_file: Optional log file path.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    if log_file:
        try:
            with open(log_file, "w"): pass
            fh = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except OSError as e:
            print(f"Warning: Could not configure file logging: {e}")

    return logger