from rgiant.preprocessing.graphs import label_pyg_data
import csv
import os
import torch
# Path to your CSV
labels_dir = r'c:\Users\piete\Documents\Development\Projects\R-GIANT\data\connectome_pipeline\labels.csv'
graphs_dir = r'c:\Users\piete\Documents\Development\Projects\R-GIANT\data\connectome_pipeline\graphs'
labeled_graphs_dir = r'c:\Users\piete\Documents\Development\Projects\R-GIANT\data\connectome_pipeline\binary_graphs'
os.makedirs(labeled_graphs_dir, exist_ok=True)

with open(labels_dir, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for session_label, diagnosis_label in reader:
        # session_label is 'pppp_ssss'
        patient_id, session_id = session_label.split('_')
        diagnosis_label = int(diagnosis_label)
        print(f"patient_id={patient_id}, session_id={session_id}, label={diagnosis_label}")

        # build path to the saved .pt file
        graph_path = os.path.join(graphs_dir, f"{patient_id}_{session_id}_G.pt")

        # load the PyG Data object
        pyg_data = torch.load(graph_path)

        if diagnosis_label > 1:
            diagnosis_label = 1

        # now apply your label function
        pyg_data = label_pyg_data(pyg_data, diagnosis_label)

        output_path = os.path.join(labeled_graphs_dir, f"{patient_id}_{session_id}_G.pt")
        torch.save(pyg_data, output_path)

