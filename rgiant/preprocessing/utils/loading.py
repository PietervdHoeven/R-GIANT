import json
import os
import yaml
from pathlib import Path

def load_parcellation_mappings():
    # Navigate to the mappings file
    mappings_path = os.path.join(os.path.dirname(__file__), "parcellation_mappings.json")

    with open(mappings_path, "r") as f:
        data = json.load(f)

    return data

def load_special_fs_labels():
    # Navigate to the special labels file
    special_labels_path = os.path.join(os.path.dirname(__file__), "special_fs_labels.json")

    with open(special_labels_path, "r") as f:
        data = json.load(f)

    return data

def load_patient_session_ids():
    """
    Load a list of session dictionaries from a JSON file.
    
    The JSON file should be an array of objects with "patient_id" and "session_id" keys.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[3]  # go 2 levels up from /parent/child/script.py
    SESSIONS_FILE = PROJECT_ROOT / "data" / "sessions.json"
    with open(SESSIONS_FILE, 'r') as f:
        sessions = json.load(f)
    return sessions

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)