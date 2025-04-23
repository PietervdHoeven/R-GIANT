from rgiant.preprocessing.nodes import extract_node_features
from rgiant.utils.loading import load_parcellation_mappings, load_special_fs_labels

session_ids_file = "data/connectome_pipeline/final_sessions.txt"
data_dir = "data/connectome_pipeline/"

mappings = load_parcellation_mappings()
special_labels = load_special_fs_labels()

print("RUNNING")

with open(session_ids_file) as f:
    labels = [label.strip() for label in f]

for label in labels:
    print(f"Extracting session {label}")
    try:
        patient_id, session_id = label.split("_", 1)
    except ValueError:
        print(f"problem with label: {label}")

    extract_node_features(
        patient_id=patient_id,
        session_id=session_id,
        data_dir=data_dir,
        mappings=mappings,
        special_labels=special_labels
    )
