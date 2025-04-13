import json
import os
from pathlib import Path
from src.preprocessing.cleaning import run_cleaning_pipeline  # Ensure cleaning.py is in your PYTHONPATH or same directory
from src.utils.logging import setup_central_logger


def load_sessions(sessions_file: str):
    """
    Load a list of session dictionaries from a JSON file.
    
    The JSON file should be an array of objects with "patient_id" and "session_id" keys.
    """
    with open(sessions_file, 'r') as f:
        sessions = json.load(f)
    return sessions

def main():
    # Setup a central logger for batch processing
    central_logger = setup_central_logger()
    central_logger.info("Starting batch processing for DWI cleaning pipeline")

    # Get the absolute path to the project root (R-GIANT/)
    ROOT = Path(__file__).resolve().parents[2]  # because src/batch_scripts/file.py → go 2 levels up
    DATA_DIR = os.path.join(ROOT, "data")

    # Process each session
    for session_dir in os.listdir(os.path.join(DATA_DIR, "mr")):
        if session_dir == ".gitkeep":
            continue
        ids = session_dir.split("_")
        patient_id = ids[0]
        session_id = ids[1]
        
        if not (patient_id and session_id):
            central_logger.warning("Invalid entry in session file: missing patient_id or session_id. Skipping entry.")
            continue
        
        try:
            central_logger.info(f"Processing patient {patient_id} | session {session_id}")
            # Run the cleaning pipeline for the current session
            # The base_path parameter is left as default ("data/"), adjust if necessary
            run_cleaning_pipeline(patient_id=patient_id, session_id=session_id, external_logger=central_logger, base_path=DATA_DIR)
        except Exception as e:
            # Log any exception without interrupting the batch processing of other sessions
            central_logger.exception(f"Error processing patient {patient_id} | session {session_id}: {e}")

if __name__ == "__main__":
    main()
