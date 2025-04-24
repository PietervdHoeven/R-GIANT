import numpy as np
from numpy.typing import NDArray
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import Compose

#TODO: Create normalisation pipelines for streamline count metrics (1log, possibly clip, z-score) and the FA and length metrics (just z-score and possibly clip)
#TODO: Implement the node feature normalisation like earlier but maybe with clipping if necessary. Not going to investigate all 98 different ROIs for outliers.

def get_transforms(
        dataset: InMemoryDataset,
        diff_p_min: float = 0.01,
        diff_p_max: float = 0.01,
        diff_clip: float = 5.0,
        diff_cols: list[int] = [0,4,6],
):
    diff_means, diff_stds, diff_qs_low, diff_qs_high = compute_diffusivity_stats(
        dataset=dataset, p_min=diff_p_min, p_max=diff_p_max, cols=diff_cols
    )

    transforms = Compose([
        NormaliseDiffusivityAttributes(
            means=diff_means,
            stds=diff_stds,
            qs_low=diff_qs_low,
            qs_high=diff_qs_high,
            clip_bound=diff_clip,
            cols=diff_cols
        )
    ])

    return transforms



def compute_diffusivity_stats(
    dataset: InMemoryDataset,
    p_min: float = 0.01,
    p_max: float = 1,
    cols: list[int] = [0, 4, 6],
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """
    Compute robust normalisation statistics for selected diffusivity metrics
    in the edge attributes of a graph dataset.

    This function extracts the specified columns from all edge attributes across
    the dataset, clips extreme tails based on quantile thresholds (Winsorisation),
    and computes the mean and standard deviation of the clipped data. These
    statistics can then be used to Z-score the same columns during preprocessing.

    Parameters
    ----------
    dataset : InMemoryDataset
        The dataset containing PyG Data objects with `edge_attr` fields.
        Typically this should be the training subset only.
    p : float, default=0.01
        The tail proportion to clip for each metric. Values below the `p`-quantile
        and above the `1-p`-quantile will be clipped.
    cols : list of int, default=[0, 4, 6]
        Column indices of `edge_attr` to compute statistics for. These typically
        correspond to diffusivity metrics known to contain extreme values or noise.

    Returns
    -------
    means : np.ndarray of shape (len(cols),)
        The mean value of each selected metric after clipping.
    stds : np.ndarray of shape (len(cols),)
        The standard deviation of each selected metric after clipping.
    qs_low : np.ndarray of shape (len(cols),)
        The lower clipping bounds (quantile `p`) for each selected metric.
    qs_high : np.ndarray of shape (len(cols),)
        The upper clipping bounds (quantile `1-p`) for each selected metric.
    """
        
    # concat each individual graph's edge attributes [E{i},7] and concat into [E_total,7]
    edge_attrs = np.concatenate([data.edge_attr.numpy() for data in dataset], axis=0)   # [E_total,7]

    # Get only the edges we care about from the specified columns 
    diff_attrs = edge_attrs[:, cols]    # [E_total, 3]

    # Compute the qunatiles that fall outside the p percentile threshold
    qs_low = np.quantile(diff_attrs, p_min, axis=0)     # (3,)
    qs_high = np.quantile(diff_attrs, 1-p_max, axis=0)  # (3,)

    # Clip the diffusivity attributes such that the tails are compressed to fall within the calculates quantiles
    clipped_attrs = np.clip(a=diff_attrs, a_min=qs_low, a_max=qs_high)  # [E_total, 3]

    # Compute the means and stds for each diffusivity metric
    means = clipped_attrs.mean(axis=0)  # (3,)
    stds = clipped_attrs.std(axis=0, ddof=0)    # (3,)

    return means, stds, qs_low, qs_high



class NormaliseDiffusivityAttributes:
    def __init__(self,
                 means: NDArray[np.float32],
                 stds: NDArray[np.float32],
                 qs_low: NDArray[np.float32],
                 qs_high: NDArray[np.float32],
                 clip_bound: float = 5.0,
                 cols: list[int] = [0,4,6]
                 ):
        self.means = torch.tensor(means).view(1, -1).float()
        self.stds = torch.tensor(stds).view(1, -1).float().clamp(min=1e-12)
        self.qs_low = torch.tensor(qs_low).view(1, -1).float()
        self.qs_high = torch.tensor(qs_high).view(1, -1).float()
        self.clip_bound = float(clip_bound)
        self.cols = cols

    def __call__(self, data: Data):
        # Get just the diffusivity columns that we want to normalise
        diff_attr = data.edge_attr[:, self.cols]    # [E, 3]

        # Winsorise clip such that tails get compressed into the p percentile limit calculated earlier
        diff_attr = torch.maximum(diff_attr, self.qs_low)
        diff_attr = torch.minimum(diff_attr, self.qs_high)  # [E, 3]

        # Compute the Z-score for the diffusivity metrics
        diff_attr = (diff_attr - self.means) / self.stds

        # write everything back into the graph data
        data.edge_attr[:, self.cols] = diff_attr

        return data

        
        






