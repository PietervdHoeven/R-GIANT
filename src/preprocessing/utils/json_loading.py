import json
import os

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