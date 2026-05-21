# core/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os

def setup_production_logger(module_name: str) -> logging.Logger:
    """Configures a thread-safe, rotating log file engine with zero parent propagation overlap."""
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # FIXED: Prevents double logging duplication artifacts inside parent standard handlers
    logger.propagate = False
    
    if not logger.handlers:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "platform.log"), 
            maxBytes=5_000_000, 
            backupCount=5,
            encoding="utf-8"
        )
        
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d]: %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger