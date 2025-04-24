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
        one_hot = x[:, :self.one_hot_dims]
        numeric = x[:, self.one_hot_dims:]

        # Extract corresponding stats
        mean_num = self.means[:, self.one_hot_dims:]
        std_num = self.stds[:, self.one_hot_dims:]

        # Z-score numeric features
        numeric_z = (numeric - mean_num) / (std_num + 1e-6)

        # Reassemble features: one-hot unchanged, numeric replaced
        data.x = torch.cat([one_hot, numeric_z], dim=1)
        return data


class ClipEdgeAttrs:
    """Clamp all edge attributes to [low, high] in-place"""
    def __init__(self, low: float = -5.0, high: float = 5.0):
        self.low = low
        self.high = high

    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_attr'):
            data.edge_attr.clamp_(self.low, self.high)
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


# ------------------------------------------------------------------------
# Node feature stats
# ------------------------------------------------------------------------

def compute_feature_stats_per_node(
    dataset: InMemoryDataset,
    one_hot_dims: int = 3,
    p_winsor: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-ROI, per-feature mean & std ignoring zeros in numeric dimensions,
    with optional winsorisation of extremes.

    Returns
    -------
    means: Tensor[N, F]
    stds : Tensor[N, F] (clamped >=1e-6; one-hot dims set to (0,1))
    """
    all_x = torch.stack([g.x for g in dataset], dim=0)  # [G, N, F]
    G, N, F = all_x.shape

    # initialize: one-hot dims => mean=0, std=1
    means = torch.zeros(N, F, dtype=all_x.dtype)
    stds = torch.ones(N, F, dtype=all_x.dtype)

    # numeric part
    numeric = all_x[:, :, one_hot_dims:]  # [G, N, F-hot]
    mask = numeric != 0                    # [G, N, F-hot]
    _, _, D = numeric.shape

    # iterate per ROI and feature
    for i in range(N):
        for d in range(D):
            col = numeric[:, i, d]
            pres = mask[:, i, d]
            vals = col[pres]
            if vals.numel() == 0:
                continue
            # winsorise
            lo = vals.quantile(p_winsor)
            hi = vals.quantile(1.0 - p_winsor)
            clipped = vals.clamp(lo, hi)
            mean = clipped.mean()
            std = clipped.std(unbiased=False).clamp(min=1e-6)
            means[i, one_hot_dims + d] = mean
            stds[i, one_hot_dims + d] = std

    return means, stds


# ------------------------------------------------------------------------
# Edge attribute stats
# ------------------------------------------------------------------------

def _winsorise(t: torch.Tensor, p: float = 0.05) -> torch.Tensor:
    """
    Clamp tensor `t` to the [p, 1-p] quantile range in-place.
    """
    lo = t.quantile(p)
    hi = t.quantile(1.0 - p)
    return t.clamp_(lo, hi)


def compute_edge_stats_per_metric(
    dataset: InMemoryDataset,
    p_winsor: float = 0.05,
    ignore_zeros: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean & std per edge-attribute metric column,
    optionally ignoring zeros and winsorising extremes.

    Returns
    -------
    means: Tensor[M]
    stds:  Tensor[M] (clamped >=1e-6)
    """
    all_eattr = torch.cat([g.edge_attr for g in dataset], dim=0)  # [ΣE, M]
    M = all_eattr.size(1)

    means, stds = [], []
    for m in range(M):
        col = all_eattr[:, m]
        if ignore_zeros:
            col = col[col != 0]
        if col.numel() == 0:
            means.append(torch.tensor(0.0, dtype=all_eattr.dtype))
            stds.append(torch.tensor(1.0, dtype=all_eattr.dtype))
            continue
        col = _winsorise(col, p=p_winsor)
        mu = col.mean()
        sigma = col.std(unbiased=False).clamp(min=1e-6)
        means.append(mu)
        stds.append(sigma)

    return torch.stack(means), torch.stack(stds)


class ZScoreEdgeAttr:
    """
    Z-score each column of data.edge_attr: (e - mean) / std
    """
    def __init__(self, means: torch.Tensor, stds: torch.Tensor):
        # means, stds: shape [M]
        self.means = means
        self.stds = stds

    def __call__(self, data: Data) -> Data:
        data.edge_attr = (data.edge_attr - self.means) / (self.stds + 1e-6)
        return data
    

class Log1pEdgeAttr:
    """
    Apply log-compression to selected edge-attribute columns:
        e  ->  sign(e) * log1p(|e|)
    Pass `cols=None` to log-transform all columns.
    """
    def __init__(self, cols: List[int] = [0,1,4,5,6]):
        self.cols = cols            # e.g. [1, 5]

    def __call__(self, data: Data) -> Data:
        ea = data.edge_attr
        idx = range(ea.size(1)) if self.cols is None else self.cols
        print(f"edge idxs {idx}")
        ea[:, idx] = torch.log1p(ea[:, idx].abs()) * ea[:, idx].sign()
        return data


def get_zscore_transform(
    dataset: InMemoryDataset,
    one_hot_dims: int = 3,
    p_node: float = 0.01,
    p_edge: float = 0.05,
    clip_bound: float = 5.0
) -> Callable[[Data], Data]:
    """
    Build a composite pre_transform that:
      1) Z-scores node features per ROI (ignoring one-hot dims)
      2) Z-scores edge-attribute metrics per column
      3) Optionally clamps edge weights to ±clip_bound
    """
    means_x, stds_x = compute_feature_stats_per_node(
        dataset, one_hot_dims=one_hot_dims, p_winsor=p_node)
    means_e, stds_e = compute_edge_stats_per_metric(
        dataset, p_winsor=p_edge)

    return Compose([
        ZScoreFeaturesPerROI(means_x, stds_x, one_hot_dims),
        ZScoreEdgeAttr(means_e, stds_e),
        ClipEdgeAttrs(-clip_bound, clip_bound)
    ])
