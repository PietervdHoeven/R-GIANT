import matplotlib.pyplot as plt
import numpy as np
from rgiant.data.dataset import ConnectomeDataset

# === Load your processed dataset ===
dataset = ConnectomeDataset(
    root="data/model_input/",
    transform=None,
    pre_transform=None
)

# === Prepare containers for each node type and its relevant feature indices ===
feature_indices = {
    "sub_cortical": [3, 9],
    "cortical":     [4, 5, 6, 7, 8, 9],
    "fluid":        [3, 9]
}

# Initialize lists
data_by_type = {nt: {fi: [] for fi in idxs} for nt, idxs in feature_indices.items()}

# === Aggregate feature values per node type ===
for g in dataset:
    x = g.x.numpy()  # shape [N, F]
    # Boolean masks for each node type
    mask_subc  = x[:, 0] == 1
    mask_cort  = x[:, 1] == 1
    mask_fluid = x[:, 2] == 1
    
    # Collect features
    for fi in feature_indices["sub_cortical"]:
        data_by_type["sub_cortical"][fi].extend(x[mask_subc, fi])
    for fi in feature_indices["cortical"]:
        data_by_type["cortical"][fi].extend(x[mask_cort, fi])
    for fi in feature_indices["fluid"]:
        data_by_type["fluid"][fi].extend(x[mask_fluid, fi])

# === Plot histograms ===
for node_type, feats in data_by_type.items():
    for fi, values in feats.items():
        # Filter out zeros before plotting non-existent-feature padding
        vals = np.array(values)
        nonzero_vals = vals[vals != 0]
        
        plt.figure()
        plt.hist(nonzero_vals, bins=50)
        plt.xlabel(f"Feature index {fi}")
        plt.ylabel("Count")
        plt.title(f"{node_type.replace('_',' ').title()} — Feature {fi} Distribution")
        plt.tight_layout()
        plt.show()