import json
import os
import logging
import time
from src.preprocessing.connectomes import run_connectome_pipeline  # Ensure cleaning.py is in your PYTHONPATH or same directory
from src.utils.logging import setup_central_logger  # Ensure logging.py is in your PYTHONPATH or same directory
from src.utils.json_loading import load_patient_session_ids, load_parcellation_mappings, load_special_fs_labels  # Ensure json_loading.py is in your PYTHONPATH or same directory


def main():
    # Load all sessions once
    sessions = load_patient_session_ids()
    mappings = load_parcellation_mappings()
    special_fs_labels = load_special_fs_labels()
    
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
            run_connectome_pipeline(patient_id=patient_id, session_id=session_id, external_logger=central_logger, fs2reduced=mappings['fs2reduced'], special_fs_labels=special_fs_labels)
            central_logger.info(f"Successfully processed patient {patient_id} | session {session_id}")
        except Exception as e:
            # Log any exception without interrupting the batch processing of other sessions
            central_logger.exception(f"Error processing patient {patient_id} | session {session_id}: {e}")

if __name__ == "__main__":
    main()
