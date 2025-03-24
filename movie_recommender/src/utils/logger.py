import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = __name__,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name (str): Name of the logger
        log_file (Optional[str]): Path to the log file. If None, only console logging is enabled
        level (int): Logging level
        format (str): Log message format
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance. If it doesn't exist, create a new one.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
