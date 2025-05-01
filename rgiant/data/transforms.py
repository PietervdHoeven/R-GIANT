import numpy as np
from numpy.typing import NDArray
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import Compose, BaseTransform

#TODO: Create normalisation pipelines for streamline count metrics (1log, possibly clip, z-score) and the FA and length metrics (just z-score and possibly clip)
#TODO: Implement the node feature normalisation like earlier but maybe with clipping if necessary. Not going to investigate all 98 different ROIs for outliers.

def get_transforms(
        dataset: InMemoryDataset,
        diff_cols: list[int] = [0,4,6],
        diff_p_min: float = 0.001,
        diff_p_max: float = 0.02,
        fl_cols: list[int] = [2,3],
        count_cols: list[int] = [1],
        count_scaling: int = 1e7,
        rel_count_cols: list[int] = [5],
        rel_count_scaling: int = 1e8,
        one_hot_dims: int = 3
        
):
    # -------------------Normalising the edge attributes----------------------------
    # Initialise containers ready for saving the computed zscoring metrics
    num_features = len(diff_cols) + len(fl_cols) + len(count_cols) + len(rel_count_cols)

    # Prepare empty containers for mean/std/quantiles across all features
    edge_means = np.zeros(num_features, dtype=np.float32)
    edge_stds = np.ones(num_features, dtype=np.float32)
    qs_low = np.zeros(num_features, dtype=np.float32)
    qs_high = np.zeros(num_features, dtype=np.float32)

    # Compute stats for FA and streamline length (no clipping)
    fl_means, fl_stds, _, _ = compute_edge_stats(dataset, cols=fl_cols)
    edge_means[fl_cols] = fl_means
    edge_stds[fl_cols] = fl_stds

    # Compute stats for diffusivity metrics (with clipping)
    diff_means, diff_stds, diff_qs_low, diff_qs_high = compute_edge_stats(
        dataset, cols=diff_cols, p_min=diff_p_min, p_max=diff_p_max
    )
    edge_means[diff_cols] = diff_means
    edge_stds[diff_cols] = diff_stds
    qs_low[diff_cols] = diff_qs_low
    qs_high[diff_cols] = diff_qs_high

    # First we log transform the streamline counts
    log_count_dataset = get_log_transformed_dataset(
        dataset=dataset, cols=count_cols, scale=count_scaling
        )
    # Compute stats for log-transformed streamline counts (no clipping)
    count_means, count_stds, _, _ = compute_edge_stats(
        log_count_dataset, cols=count_cols
        )
    edge_means[count_cols] = count_means
    edge_stds[count_cols] = count_stds

    # We do this for different columns because we need to scale the values before we can log transform them
    log_count_dataset = get_log_transformed_dataset(
        dataset=dataset, cols=rel_count_cols, scale=rel_count_scaling
        )   
    # Compute stats for log-transformed normalised streamline counts (with clipping)
    norm_count_means, norm_count_stds, _, _ = compute_edge_stats(
        log_count_dataset, cols=rel_count_cols
        )
    edge_means[rel_count_cols] = norm_count_means
    edge_stds[rel_count_cols] = norm_count_stds

    # -----------------------Normalising the node features------------------------
    # Compute the stats necessary for initialising the node feature normalisation class
    node_means, node_stds = compute_node_stats(dataset=dataset, one_hot_dims=one_hot_dims)


    # Final Compose pipeline
    transforms = Compose([
        LogEdgeAttributes(
            cols=count_cols,
            scale=count_scaling
            ),
        LogEdgeAttributes(
            cols=rel_count_cols,
            scale=rel_count_scaling
        ),
        ZscoreClipEdgeAttributes(
            means=edge_means,
            stds=edge_stds,
            qs_low=qs_low,
            qs_high=qs_high,
        ),
        ZScoreFeaturesPerROI(
            means=node_means,
            stds=node_stds
        )
    ])

    return transforms



def compute_edge_stats(
    dataset: InMemoryDataset,
    cols: list[int],
    p_min: float = 0.0,
    p_max: float = 0.0,
):
    """
    Compute robust normalization statistics for selected edge attributes.

    This function iterates over all graphs in the given PyG InMemoryDataset,
    extracts the specified columns from each `edge_attr`, applies percentile-based
    clipping (Winsorization) according to `p_min` and `p_max`, and then computes
    the mean and standard deviation of the clipped values.

    Parameters
    ----------
    dataset : InMemoryDataset
        A PyG InMemoryDataset containing `Data` objects with an `edge_attr` field.
    cols : list[int]
        Indices of the edge attribute columns to process.
    p_min : float, default=0.0
        Lower-tail clipping quantile (0 ≤ p_min < 1). Values below this quantile
        are set to the `p_min` threshold.
    p_max : float, default=0.0
        Upper-tail clipping quantile (0 ≤ p_max < 1). Values above the
        `1 - p_max` quantile are set to the corresponding threshold.

    Returns
    -------
    means : np.ndarray, shape (len(cols),)
        The mean of each selected column after clipping.
    stds : np.ndarray, shape (len(cols),)
        The standard deviation (population, ddof=0) of each clipped column.
    qs_low : np.ndarray, shape (len(cols),)
        The lower quantile thresholds used for clipping (p_min).
    qs_high : np.ndarray, shape (len(cols),)
        The upper quantile thresholds used for clipping (1 - p_max).
    """
        
    # concat each individual graph's edge attributes [E{i},7] and concat into [E_total,7]
    edge_attrs = np.concatenate([data.edge_attr.numpy() for data in dataset], axis=0)   # [E_total,7]

    # Get only the edges we care about from the specified columns 
    diff_attrs = edge_attrs[:, cols]    # [E_total, len(cols)]

    # Compute the qunatiles that fall outside the p percentile threshold
    qs_low = np.quantile(diff_attrs, p_min, axis=0)     # (len(cols),)
    qs_high = np.quantile(diff_attrs, 1-p_max, axis=0)  # (len(cols),)

    # Clip the diffusivity attributes such that the tails are compressed to fall within the calculates quantiles
    clipped_attrs = np.clip(a=diff_attrs, a_min=qs_low, a_max=qs_high)  # [E_total, len(cols)]

    # Compute the means and stds for each diffusivity metric
    means = clipped_attrs.mean(axis=0)  # (len(cols),)
    stds = clipped_attrs.std(axis=0, ddof=0)    # (len(cols),)

    return means, stds, qs_low, qs_high



class ZscoreClipEdgeAttributes(BaseTransform):
    def __init__(self,
                 means: NDArray[np.float32],
                 stds: NDArray[np.float32],
                 qs_low: NDArray[np.float32] = np.zeros(7,),
                 qs_high: NDArray[np.float32] = np.zeros(7,)
                 ):
        super().__init__()

        self.means = torch.tensor(means).view(1, -1).float()
        self.stds = torch.tensor(stds).view(1, -1).float().clamp(min=1e-12)

        self.qs_low = torch.tensor(qs_low).view(1, -1).float()
        self.qs_high = torch.tensor(qs_high).view(1, -1).float()

        # clipping mask for columns that have actual values to clip and not just 0s
        self.clip_mask = (self.qs_high != self.qs_low)

    def __call__(self, data: Data):
        # get data
        edge_attr = data.edge_attr

        # Winsorise clip such that tails get compressed into the p percentile limit calculated earlier
        if self.clip_mask.any():
            edge_attr[:, self.clip_mask[0]] = torch.maximum(
                edge_attr[:, self.clip_mask[0]],
                self.qs_low[:, self.clip_mask[0]]
            )
            edge_attr[:, self.clip_mask[0]] = torch.minimum(
                edge_attr[:, self.clip_mask[0]],
                self.qs_high[:, self.clip_mask[0]]
            )

        # Compute the Z-score for the diffusivity metrics
        edge_attr = (edge_attr - self.means) / self.stds

        # write everything back into the graph data
        data.edge_attr = edge_attr

        return data

        
        
def get_log_transformed_dataset(dataset: InMemoryDataset, cols: list[int], scale: float = 1.0):
    log_dataset = []
    for data in dataset:
        data = data.clone()
        data.edge_attr[:, cols] = torch.log1p(data.edge_attr[:, cols] * scale)  # actual log transform
        log_dataset.append(data)
    return log_dataset



class LogEdgeAttributes(BaseTransform):
    def __init__(self, cols: list[int], scale: float = 1.0):
        super().__init__()
        self.cols = cols
        self.scale = scale
    
    def __call__(self, data):
        data.edge_attr[:, self.cols] = torch.log1p(data.edge_attr[:, self.cols] * self.scale)
        return data



def compute_node_stats(
        dataset: InMemoryDataset,
        one_hot_dims: int = 3
):
    x_list = []
    for data in dataset:    # for each graph
        x_num = data.x[:, one_hot_dims:].numpy()    # (N, F_num) Only take the numerical node features, not the one-hot features
        x_num = np.where(x_num == 0.0, np.nan, x_num)   # Replace exact 0 values with NaN so we can ignore them when taking mean and std
        x_list.append(x_num)    # Save to list

    x_stacked = np.stack(x_list, axis=0)    # (G, N, F_num)

    means = np.nanmean(x_stacked, axis=0)   # (N, F_num)
    stds = np.nanstd(x_stacked, axis=0)     # (N, F_num)

    # Replace nans with neutral values
    means = np.nan_to_num(means, nan=0.0)
    stds = np.nan_to_num(stds, nan=1.0)

    return means, stds



class ZScoreFeaturesPerROI(BaseTransform):
    def __init__(self, means: np.ndarray, stds: np.ndarray, one_hot_dims: int = 3):
        super().__init__()
        self.means = torch.tensor(means).float()    # [N, F_num]
        self.stds  = torch.tensor(stds).float().clamp(min=1e-12)    # [N, F_num]
        self.one_hot_dims = one_hot_dims

    def __call__(self, data: Data) -> Data:
        x = data.x  # [N, F]
        x_num = x[:, self.one_hot_dims:]    # [N, F_num]
        x[:, self.one_hot_dims:] = (x_num - self.means) / self.stds
        return data

