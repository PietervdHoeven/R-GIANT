import os
import time
import logging

def setup_central_logger():
    """
    Sets up a centralized logger to capture log messages for all sessions.
    All logs are saved to a centralized log file.
    """
    logger = logging.getLogger("batch_cleaning_logger")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if logger is recreated in subsequent runs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create the logs directory if it doesn't exist
    log_dir = os.path.join("logs", "batch")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a central log file with a timestamp in the name
    timestamp = time.strftime("%d%m%Y_%H%M%S")
    log_filepath = os.path.join(log_dir, f"batch_cleaning_{timestamp}.log")
    
    # Create a file handler for the central logger
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Optional: add a stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger