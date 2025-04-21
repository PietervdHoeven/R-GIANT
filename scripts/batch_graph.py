from rgiant.preprocessing.graphs import build_pyg_data

session_ids_file = "data/final_sessions.txt"
data_dir = "data/"

print("RUNNING")

with open(session_ids_file) as f:
    labels = [label.strip() for label in f]

for label in labels:
    print(f"building garph for {label}")
    try:
        patient_id, session_id = label.split("_", 1)
    except ValueError:
        print(f"problem with label: {label}")

    build_pyg_data(
        patient_id=patient_id,
        session_id=session_id,
        data_dir=data_dir
    )
