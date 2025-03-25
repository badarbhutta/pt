import logging
import sys
from typing import Optional

from app.core.config import settings

# Configure logging format
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger for the specified module name.
    """
    logger = logging.getLogger(name or __name__)
    
    # Only configure logger if it hasn't been configured yet
    if not logger.handlers:
        # Set log level based on settings
        log_level = getattr(logging, settings.LOG_LEVEL.upper())
        logger.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(console_handler)
        
        # Add file handler if log file path is specified
        if settings.LOG_FILE:
            try:
                file_handler = logging.FileHandler(settings.LOG_FILE)
                file_handler.setFormatter(logging.Formatter(log_format, date_format))
                logger.addHandler(file_handler)
            except Exception as e:
                logger.error(f"Failed to create file handler for logging: {str(e)}")
        
        # Prevent propagation to the root logger
        logger.propagate = False
    
    return logger

# Initialize root logger
root_logger = get_logger("period_tracker_ml")
