import os
from pathlib import Path
from src.preprocessing.connectomes import run_connectome_pipeline  # Ensure cleaning.py is in your PYTHONPATH or same directory
from src.utils.logging import setup_central_logger  # Ensure logging.py is in your PYTHONPATH or same directory
from src.utils.json_loading import load_patient_session_ids, load_parcellation_mappings, load_special_fs_labels  # Ensure json_loading.py is in your PYTHONPATH or same directory


def main():
    # Load all sessions once
    mappings = load_parcellation_mappings()
    special_fs_labels = load_special_fs_labels()
    
    # Setup a central logger for batch processing
    central_logger = setup_central_logger()
    central_logger.info("Starting batch processing for DWI cleaning pipeline")

    # Get the absolute path to the project root (R-GIANT/)
    ROOT = Path(__file__).resolve().parents[2]  # because src/batch_scripts/file.py → go 2 levels up
    DATA_DIR = os.path.join(ROOT, "data")
    
    # Process each session
    for session_dir in os.listdir(os.path.join(DATA_DIR, "clean")):
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
            run_connectome_pipeline(patient_id=patient_id, session_id=session_id, external_logger=central_logger, fs_idxs2graph_idxs=mappings['fs_idxs2graph_idxs'], special_fs_labels=special_fs_labels, base_dir=DATA_DIR)
            central_logger.info(f"Successfully processed patient {patient_id} | session {session_id}")
        except Exception as e:
            # Log any exception without interrupting the batch processing of other sessions
            central_logger.exception(f"Error processing patient {patient_id} | session {session_id}: {e}")

if __name__ == "__main__":
    main()
