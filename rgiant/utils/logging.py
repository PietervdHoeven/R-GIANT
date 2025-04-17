import logging
import os
import time

def setup_logger(name="rgiant_logger", log_dir="logs/", prefix="task",
                         stream=True, patient_id=None, session_id=None):
    """
    Sets up a centralized logger with configurable name, output location, and filename prefix.
    The log file name will include patient/session ID if provided.

    Args:
        name (str): Logger name.
        log_dir (str): Directory where the log file will be saved.
        prefix (str): Task prefix (e.g., 'cleaning', 'connectomes').
        stream (bool): Whether to stream logs to console.
        patient_id (str, optional): Participant ID to include in log filename.
        session_id (str, optional): Session ID to include in log filename.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%d%m%Y_%H%M%S")

    filename_parts = [prefix]
    if patient_id: filename_parts.append(str(patient_id))
    if session_id: filename_parts.append(str(session_id))
    filename_parts.append(timestamp)

    log_filename = "_".join(filename_parts) + ".log"
    log_path = os.path.join(log_dir, log_filename)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

