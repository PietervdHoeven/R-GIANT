# tests/test_dataset.py

import os
import glob
import torch
import numpy as np
import pytest
from torch_geometric.data import Data

from rgiant.data.dataset import ConnectomeDataset

# point this to wherever you keep your built graphs
RAW_DIR = "data/model_input/graphs"
PROCESSED_FILE = "all_graphs.pt"

def test_connectome_dataset(root="data/model_input/"):
    # 3) Instantiate dataset (this will process and save to tmp_root/processed/)
    ds = ConnectomeDataset(root=root, processed_filename="test.pt")

    # 5) Length matches
    # count raw graph files
    raw_paths = glob.glob(os.path.join("data/model_input/raw/", "*_G.pt"))
    n_raw = len(raw_paths)
    assert len(ds) == n_raw, f"expected {n_raw} graphs, got {len(ds)}"

    # 6) Walk through every graph and do basic sanity checks
    node_counts = []
    edge_counts = []
    labels = []
    for idx in range(len(ds)):
        g = ds[idx]
        # node features
        assert hasattr(g, "x") and isinstance(g.x, torch.Tensor)
        node_counts.append(g.x.size(0))

        # edges
        assert hasattr(g, "edge_index") and g.edge_index.size(0) == 2
        E = g.edge_index.size(1)
        edge_counts.append(E)

        # multi-view fields
        assert hasattr(g, "edge_type") and isinstance(g.edge_type, torch.Tensor)
        assert hasattr(g, "edge_weight") and isinstance(g.edge_weight, torch.Tensor)
        assert g.edge_type.size(0) == E and g.edge_weight.size(0) == E

        # optional label
        if hasattr(g, "y"):
            assert isinstance(g.y, torch.Tensor) and g.y.numel() == 1
            labels.append(int(g.y.item()))

        # id matches filename
        base = os.path.basename(raw_paths[idx])
        expected_id = base.replace("_G.pt", "")
        assert g.id == expected_id

    # 7) Print a brief summary (so you can eyeball it in pytest -q output)
    print(f"\nNodes per graph: min={min(node_counts)}, max={max(node_counts)}, mean={np.mean(node_counts):.1f}")
    print(f"Edges per graph: min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}")
    if labels:
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"Label distribution: {dist}")

    print(ds[0].x.shape)

if __name__ == "__main__":
    test_connectome_dataset()
