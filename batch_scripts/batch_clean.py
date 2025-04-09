import json
import os
import logging
import time
from src.preprocessing.cleaning import run_cleaning_pipeline  # Ensure cleaning.py is in your PYTHONPATH or same directory

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

def load_sessions(sessions_file: str):
    """
    Load a list of session dictionaries from a JSON file.
    
    The JSON file should be an array of objects with "patient_id" and "session_id" keys.
    """
    with open(sessions_file, 'r') as f:
        sessions = json.load(f)
    return sessions

def main():
    # Specify the sessions JSON file (you can modify the path as needed)
    sessions_file = "data\sessions.json"
    
    # Load all sessions once
    sessions = load_sessions(sessions_file)
    
    # Setup a central logger for batch processing
    central_logger = setup_central_logger()
    central_logger.info("Starting batch processing for DWI cleaning pipeline")
    
    # Process each session
    for entry in sessions:
        patient_id = entry.get("patient_id")
        session_id = entry.get("session_id")
        
        if not (patient_id and session_id):
            central_logger.warning("Invalid entry in session file: missing patient_id or session_id. Skipping entry.")
            continue
        
        try:
            central_logger.info(f"Processing patient {patient_id} | session {session_id}")
            # Run the cleaning pipeline for the current session
            # The base_path parameter is left as default ("data/"), adjust if necessary
            run_cleaning_pipeline(patient_id=patient_id, session_id=session_id, external_logger=central_logger)
        except Exception as e:
            # Log any exception without interrupting the batch processing of other sessions
            central_logger.exception(f"Error processing patient {patient_id} | session {session_id}: {e}")

if __name__ == "__main__":
    main()
