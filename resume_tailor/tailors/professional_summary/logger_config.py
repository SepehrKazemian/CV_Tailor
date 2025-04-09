import logging

LOG_FILE = "professional_summary_refinement.log"

# Clear existing log
try:
    with open(LOG_FILE, "w"):
        pass
except OSError as e:
    print(f"Warning: Could not clear log file {LOG_FILE}: {e}")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
