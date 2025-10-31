# logger.py - ENHANCED
import logging
import sys
from datetime import datetime
import os

def setup_logger():
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    logger = logging.getLogger("cv_chatbot")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(f"logs/cv_chatbot_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()