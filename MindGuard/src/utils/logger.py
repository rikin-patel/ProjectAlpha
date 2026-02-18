"""Centralized logging utility for the application."""
import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = 'logs', log_level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger that writes to both console and a file.

    Args:
        name (str): Name of the logger (typically __name__).
        log_dir (str): Directory where the log file will be saved.
        log_level: Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLoggger(name)
    if logger.handlers:
        # Logger already configured
        return logger
    
    logger.setLevel(log_level)

    # Create handlers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(os.path.join(log_dir, f'{name}_{timestamp}.log'))

    c_handler.setLevel(log_level)
    f_handler.setLevel(log_level)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger