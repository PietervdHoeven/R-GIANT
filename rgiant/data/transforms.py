# data/transforms.py
import torch
from torch_geometric.data import Data, InMemoryDataset
from typing import Callable, Tuple, List


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



def _winsorise(t: torch.Tensor, p=0.01):
    lo, hi = t.quantile(p), t.quantile(1-p)
    return t.clamp(lo, hi)



def compute_edge_stats_per_relation(
    dataset: InMemoryDataset,
    p_winsor: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-relation mean / std, after winsorising the strongest `p_winsor` tails.

    Parameters
    ----------
    dataset   : InMemoryDataset  - graphs with .edge_weight & .edge_type
    p_winsor  : float            - fraction of each tail to clamp (default 1 %)

    Returns
    -------
    means : [R] tensor
    stds  : [R] tensor   (std is clamped ≥ 1e-6 to avoid /0)
    """
    # 1) figure out how many relation IDs we have
    max_rel = 0
    for g in dataset:
        max_rel = max(max_rel, int(g.edge_type.max()))
    R = max_rel + 1

    # 2) gather edge weights per relation
    buckets = [[] for _ in range(R)]
    for g in dataset:
        w, r = g.edge_weight, g.edge_type
        for rel in r.unique().tolist():
            buckets[rel].append(w[r == rel])

    # 3) winsorise → concat → mean / std
    means, stds = [], []
    for rel, chunks in enumerate(buckets):
        vec = torch.cat(chunks, dim=0)
        vec = _winsorise(vec, p=p_winsor)
        means.append(vec.mean())
        stds .append(vec.std(unbiased=False).clamp(min=1e-6))

    return torch.stack(means), torch.stack(stds)



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
        ClipEdgeWeights(low=-4.0, high=4.0)
    ])
