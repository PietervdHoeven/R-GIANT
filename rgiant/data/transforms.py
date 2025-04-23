# data/transforms.py
import torch
from torch_geometric.data import Data, InMemoryDataset
from typing import Callable, Optional, Tuple, List


class ZScoreFeaturesPerROI:
    """
    Normalize node features per ROI index: z-score only numeric features, leave one-hot dims unchanged.
    Assumes first `one_hot_dims` columns of data.x are one-hot indicators and should not be normalized.
    """
    def __init__(self, means: torch.Tensor, stds: torch.Tensor, one_hot_dims: int = 3):
        # means, stds: shape [N, F]
        self.means = means
        self.stds = stds
        self.one_hot_dims = one_hot_dims

    def __call__(self, data: Data) -> Data:
        x = data.x  # [N, F]
        # Split into one-hot and numeric parts
        one_hot = x[:, :self.one_hot_dims]  # preserve these
        numeric = x[:, self.one_hot_dims:]

        # Extract corresponding stats
        mean_num = self.means[:, self.one_hot_dims:]
        std_num = self.stds[:, self.one_hot_dims:]

        # Z-score numeric features
        numeric_z = (numeric - mean_num) / (std_num + 1e-6)

        # Reassemble features: one-hot unchanged, numeric replaced
        data.x = torch.cat([one_hot, numeric_z], dim=1)
        return data



class ZScoreEdgeWeightsPerRelation:
    """
    Normalize edge weights per relation type: z-score using precomputed means/stds.
    """
    def __init__(self, means: torch.Tensor, stds: torch.Tensor):
        # means, stds: shape [R]
        self.means = means
        self.stds = stds

    def __call__(self, data: Data) -> Data:
        w = data.edge_weight
        r = data.edge_type
        mean_r = self.means[r]
        std_r = self.stds[r]
        data.edge_weight = (w - mean_r) / (std_r + 1e-6)
        return data



class ClipEdgeWeights:
    """Clamp all edge weights to [low, high] *in-place*."""
    def __init__(self, low: float = -5.0, high: float = 5.0):
        self.low  = low
        self.high = high

    def __call__(self, data):
        data.edge_weight.clamp_(self.low, self.high)
        return data



class Compose:
    """
    Compose multiple Data transforms into one callable.
    """
    def __init__(self, transforms: List[Callable[[Data], Data]]):
        self.transforms = transforms

    def __call__(self, data: Data) -> Data:
        for t in self.transforms:
            data = t(data)
        return data



# def compute_feature_stats_per_node(dataset: InMemoryDataset) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Compute mean and std per node (each node corresponds to a neural ROI) across all graphs in the dataset.

#     Args:
#         dataset: a PyG InMemoryDataset (no transforms applied) of length G,
#                  where each item g.x has shape [N, F].

#     Returns:
#         mean: Tensor of shape [N, F]
#         std:  Tensor of shape [N, F]
#     """
#     # 1) Collect all feature matrices
#     #    List of G tensors, each [N, F]
#     all_x = [g.x for g in dataset]

#     # 2) Stack into [G, N, F]
#     all_x = torch.stack(all_x, dim=0)

#     # 3) Compute per-ROI (i.e. along the G axis)
#     means = all_x.mean(dim=0)  # [N, F]
#     stds  = all_x.std(dim=0)   # [N, F]

#     return means, stds



def compute_feature_stats_per_node(
    dataset: InMemoryDataset,
    one_hot_dims: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-ROI, per-feature mean & std ignoring zeros in the numeric part.

    Returns
    -------
    means : [N, F]
    stds  : [N, F]   (clamped ≥1e-6; one-hot dims forced to (mean=0, std=1))
    """
    all_x = torch.stack([g.x for g in dataset], dim=0)   # [G, N, F]
    N, F  = all_x.shape[1:]

    # one-hot part -> mean=0, std=1 so they’ll stay 0/1 after transform
    means = torch.zeros(N, F, dtype=all_x.dtype)
    stds  = torch.ones (N, F, dtype=all_x.dtype)

    numeric = all_x[:, :, one_hot_dims:]                 # [G, N, F-hot]
    mask    = numeric != 0                               # present values

    # sum & count per ROI/feature, ignoring zeros
    sum_   = (numeric * mask).sum(dim=0)                    # [N, F-hot]
    count  = mask.sum(dim=0).clamp(min=1e-6)                # avoid /0
    mean_n = sum_ / count

    # variance
    var = (((numeric - mean_n) * mask) ** 2).sum(dim=0) / count
    std_n = var.sqrt().clamp(min=1e-6)

    # insert back into full tensors
    means[:, one_hot_dims:] = mean_n
    stds [:, one_hot_dims:] = std_n
    return means, stds



def compute_edge_stats_per_relation(dataset: InMemoryDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-relation mean and std of edge weights over all graphs in `dataset`.

    Args:
        dataset: a PyG InMemoryDataset whose items have:
                 - data.edge_weight: Tensor[E]
                 - data.edge_type:   Tensor[E] with values in {0,...,R-1}

    Returns:
        means: Tensor of shape [R], the mean weight for each relation
        stds:  Tensor of shape [R], the std. deviation for each relation
    """
    # 1) Infer number of relations
    max_r = 0
    for data in dataset:
        max_r = max(max_r, int(data.edge_type.max().item()))
    R = max_r + 1

    # 2) Collect weights per relation
    weights_per_rel = [[] for _ in range(R)]
    for data in dataset:
        w = data.edge_weight
        r = data.edge_type
        for rel in range(R):
            mask = (r == rel)
            if mask.any():
                weights_per_rel[rel].append(w[mask])

    # 3) Compute statistics
    means = torch.zeros(R, dtype=torch.float)
    stds  = torch.zeros(R, dtype=torch.float)
    for rel, w_list in enumerate(weights_per_rel):
        if w_list:
            all_w = torch.cat(w_list, dim=0)
            means[rel] = all_w.mean()
            stds[rel]  = all_w.std(unbiased=False)

    return means, stds



def get_zscore_transform(dataset: InMemoryDataset) -> Callable[[Data], Data]:
    """
    Build a pre_transform that applies per-ROI node-feature z-score
    and per-relation edge-weight z-score, using stats from raw_dataset.
    """
    means_x, stds_x = compute_feature_stats_per_node(dataset)
    means_w, stds_w = compute_edge_stats_per_relation(dataset)
    return Compose([
        ZScoreFeaturesPerROI(means_x, stds_x),
        ZScoreEdgeWeightsPerRelation(means_w, stds_w),
        ClipEdgeWeights(-5.0, 5.0) 
    ])
