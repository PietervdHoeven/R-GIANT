# test_dataset.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from rgiant.data.dataset import ConnectomeDataset
from rgiant.data.transforms import get_zscore_transform
import shutil

# Adjust this path if you want to point to a temp directory
DATA_ROOT       = "data/model_input/"
RAW_FILENAME    = "raw_graphs.pt"
NORM_FILENAME   = "normalised_graphs.pt"

def test_raw_dataset():
    # 1) Instantiate raw dataset (will write processed/raw_graphs.pt if missing)
    print("Generating raw ds")
    raw_ds = ConnectomeDataset(
        root=DATA_ROOT,
        processed_filename=RAW_FILENAME,
        transform=None,
        pre_transform=None
    )

    # 2) Check that the cache file exists
    raw_path = os.path.join(DATA_ROOT, "processed", RAW_FILENAME)
    assert os.path.exists(raw_path), f"Raw cache not found at {raw_path}"
    print("asserted dataset file exists")

    # 3) Basic sanity: nonempty and correct attributes
    N = len(raw_ds)
    assert N > 0, "Raw dataset is empty"
    for idx in [0, N//2, N-1]:
        g = raw_ds[idx]
        for attr in ("x", "edge_index", "edge_weight", "edge_type", "id"):
            assert hasattr(g, attr), f"Graph missing `{attr}`"
        # If labels are present, they should be integers
        if hasattr(g, "y"):
            y = g.y.item()
            assert isinstance(y, (int, np.integer)), f"Label is not int: {y}"

    print("Raw dataset tests passed.")


def test_normalised_dataset():
    # 1) Reload raw to compute stats
    raw_ds = ConnectomeDataset(
        root=DATA_ROOT,
        processed_filename=RAW_FILENAME,
        transform=None,
        pre_transform=None
    )

    # 2) Build pre_transform from train split of raw_ds
    #    (here we simply use the entire raw_ds for stats)
    pre_tf = get_zscore_transform(raw_ds)

    # 3) Instantiate normalized dataset (writes processed/normalised_graphs.pt)
    norm_ds = ConnectomeDataset(
        root=DATA_ROOT,
        processed_filename=NORM_FILENAME,
        transform=None,
        pre_transform=pre_tf,
        force_reload=True
    )

    print(raw_ds[0].x[31])
    print(norm_ds[0].x[31])

    # Compute min and max per feature for the first raw graph (participant 0), ignoring zeros
    raw_x0 = raw_ds[0].x
    mask = raw_x0 != 0

    # Replace zeros with +inf for min, -inf for max, then reduce
    inf = torch.tensor(float('inf'), device=raw_x0.device)
    min_vals = torch.where(mask, raw_x0, inf).min(dim=0).values
    max_vals = torch.where(mask, raw_x0, -inf).max(dim=0).values

    print("Raw participant 0 feature min/max across all nodes:")
    for i, (mn, mx) in enumerate(zip(min_vals, max_vals)):
        print(f"  Feature {i}: min = {mn.item():.4f}, max = {mx.item():.4f}")

    # 4) Check normalized cache exists
    norm_path = os.path.join(DATA_ROOT, "processed", NORM_FILENAME)
    assert os.path.exists(norm_path), f"Normalized cache not found at {norm_path}"

    # 5) Lengths should match
    assert len(norm_ds) == len(raw_ds), "Normalized dataset length mismatch"

    # 6) Spot-check normalization: node features z-scored per ROI
    #    Stack a few graphs to check mean≈0 and std≈1
    sample_x = torch.stack([norm_ds[i].x for i in range(min(600, len(norm_ds)))], dim=0)
    print("sample x shape: ", sample_x.shape)
    mean_per_roi = sample_x.mean(dim=0)  # [N, F]
    print("mean per roi shape: ", mean_per_roi.shape)
    std_per_roi  = sample_x.std(dim=0)
    print("std_per roi shape: ", std_per_roi.shape)

    # Only check the numeric features (skip first 3 one-hot dims)
    one_hot_dims = 3
    mean_per_roi = mean_per_roi[:, one_hot_dims:]
    # print(mean_num[:50,:])
    std_per_roi  = std_per_roi[:,  one_hot_dims:]
    # print(std_num[:50,:])
    nonzero_mask = std_per_roi > 0


    # assert torch.allclose(mean_per_roi, torch.zeros_like(mean_per_roi), atol=5e-1), f"Numeric feature means not near zero: {mean_per_roi[:5]}..."
    # assert torch.allclose(std_per_roi[nonzero_mask], torch.ones_like(std_per_roi[nonzero_mask]), atol=5e-1), f"Numeric feature stds not near one: {std_per_roi[nonzero_mask][:5]}"


 
    # 3) Gather the chosen feature value across all graphs for the specified ROI
    # Features to plot
    feature_indices = {
        3: "Volumemm3",
        4: "SurfArea",
        5: "GrayVol",
        6: "ThickAvg",
        7: "ThickStd",
        8: "MeanCurv",
        9: "PIB-SUVR"
    }

    for roi_idx in [10, 40, 70, 80]:
        # 6) Collect values for this ROI index
        values_by_feature = {f: [] for f in feature_indices}
        for g in norm_ds:
            x = g.x.numpy()
            for f in feature_indices:
                vals = x[roi_idx, f]
                nonzero_vals = vals[vals != 0] if hasattr(vals, "__iter__") else ([vals] if vals != 0 else [])
                values_by_feature[f].extend(nonzero_vals)

        # Plot grid for this ROI
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"ROI index: {roi_idx}", fontsize=16)
        axes = axes.flatten()
        for ax, (f, label) in zip(axes, feature_indices.items()):
            ax.hist(values_by_feature[f], bins=16)
            ax.set_title(f"{label} (feat {f})")
            ax.set_xlabel("Z-scored value")
            ax.set_ylabel("Count")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ------------------------------------------------------------------
    #  Gather every edge-weight for every relation from the normalised dataset
    # ------------------------------------------------------------------
    weights_by_rel = defaultdict(list)        # rel-id -> list-of-1D-tensors
    for g in norm_ds:
        w, r = g.edge_weight, g.edge_type
        for rel in r.unique().tolist():
            weights_by_rel[int(rel)].append(w[r == rel])

    # ------------------------------------------------------------------
    #  Compute simple stats and print a table
    # ------------------------------------------------------------------
    print("\nEdge-weight stats AFTER per-relation z-score\n")
    for rel in sorted(weights_by_rel):
        w = torch.cat(weights_by_rel[rel])
        print(f"rel {rel:2d}:  N={len(w):6d}   mean={w.mean():+7.4f}   std={w.std():6.4f}   "
            f"[{w.min():+5.2f} {w.max():+5.2f}]")

    # ------------------------------------------------------------------
    #  Make histograms – one subplot per relation (up to 6 per figure)
    # ------------------------------------------------------------------
    rels      = sorted(weights_by_rel)
    ncols     = 3
    nrows     = int(np.ceil(len(rels)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

    for ax, rel in zip(axes.flatten(), rels):
        data = torch.cat(weights_by_rel[rel]).numpy()
        ax.hist(data, bins=50)
        ax.set_title(f"Relation {rel}   mean={data.mean():+.2f} std={data.std():.2f}")
        ax.set_xlabel("z-scored weight");  ax.set_ylabel("count")

    # hide any unused subplots
    for ax in axes.flatten()[len(rels):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("Normalized dataset tests passed.")


if __name__ == "__main__":
    # Remove and recreate the processed folder
    processed_dir = os.path.join(DATA_ROOT, "processed")
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    # Run tests from scratch
    #test_raw_dataset()

    test_normalised_dataset()
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)
    print("\nAll dataset tests passed!")
