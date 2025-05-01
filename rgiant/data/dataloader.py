# dataloader.py
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Import your graph dataset class and normalization transforms
from rgiant.data.dataset import ConnectomeDataset
from rgiant.data.transforms import get_transforms



def compute_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.
    """
    classes = np.unique(labels)
    print(f"classes: {classes}")
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)



def make_split_loaders(
    dataset_root: str,
    batch_size: int,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 19,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = None
):
    """
    Create train/val/test DataLoaders with a pre_transform based on train-set stats.
    Returns: dict with loaders and class_weights
    """
    # 1) Initial load to get labels and indices (no transforms applied)
    raw_dataset = ConnectomeDataset(
        root=dataset_root,
        processed_filename="raw_graphs.pt",
        transform=None,
        pre_transform=None
    )
    labels = np.array([data.y.item() for data in raw_dataset])  # array containing all the target labels
    indices = np.arange(len(raw_dataset))   # Array for all the indices of each graph

    # 2) Split indices: We first splinter off the test subset and then splinter off the validation subset
    idx_train, idx_test, y_train, _ = train_test_split(
        indices, labels,
        test_size=test_size,
        stratify=labels,    # We split based on the labels. So 10% of the graphs will contain a same ratio HC:MCI/AD
        random_state=random_state
    )
    val_size = val_size / (1 - test_size) # Update the val_size to correct for the remaining datapoints after the first split
    idx_train, idx_val, y_train, _ = train_test_split(
        idx_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state
    )

    # 3) Calculate the transformation function by passing the dataset and letting transforms.py calculate behind the scenes
    raw_subset = Subset(raw_dataset, idx_train) # Pass only the training part of the raw data to determine the transformation function
    pre_transform = get_transforms(dataset=raw_subset)   # We therefore determine how to normalise by only looking at the training data

    # 4) Reinstantiate dataset with pre_transform applied at processing
    normalised_dataset = ConnectomeDataset(
        root=dataset_root,
        processed_filename="normalised_graphs.pt",
        transform=None,
        pre_transform=pre_transform,
        force_reload=True
    )

    # 5) Then take subsets determined by the splits we made earlier
    train_ds = Subset(normalised_dataset, idx_train)
    val_ds   = Subset(normalised_dataset, idx_val)
    test_ds  = Subset(normalised_dataset, idx_test)

    # 6) Build DataLoaders
    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)  # We only shuffle the train_loader so that training is more random
    val_loader   = DataLoader(val_ds, shuffle=False, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

    # 7) Compute class weights on train labels
    class_weights = compute_weights(y_train)    # These are necessary for us to alter the loss function to manage the class inbalance

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'class_weights': class_weights
    }

# def make_cv_loaders(
#     dataset_root: str,
#     batch_size: int,
#     n_splits: int = 5,
#     random_state: int = 19,
#     num_workers: int = 4,
#     pin_memory: bool = True,
#     persistent_workers: bool = True,
#     prefetch_factor: int = 2,
#     normalization: bool = True
# ):
#     """
#     Yield cross-validation DataLoaders and class weights per fold.
#     Yields dicts with train_loader, val_loader, class_weights
#     """
#     # Get transforms
#     train_tf, val_tf, _ = get_normalization_transforms() if normalization else (None, None, None)

#     # Load full dataset
#     full_dataset = ConnectomeDataset(
#         root=dataset_root,
#         transform=None,
#         pre_transform=None
#     )
#     labels = np.array([data.y.item() for data in full_dataset])
#     indices = np.arange(len(full_dataset))

#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#     for train_idx, val_idx in skf.split(indices, labels):
#         # Subsets and transforms
#         train_ds = Subset(full_dataset, train_idx)
#         train_ds.dataset.transform = train_tf
#         val_ds   = Subset(full_dataset, val_idx)
#         val_ds.dataset.transform = val_tf

#         dl_kwargs = dict(
#             batch_size=batch_size,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             persistent_workers=persistent_workers,
#             prefetch_factor=prefetch_factor
#         )

#         train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
#         val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)

#         class_weights = compute_weights(labels[train_idx])

#         yield {
#             'train_loader': train_loader,
#             'val_loader': val_loader,
#             'class_weights': class_weights
#         }


# def get_dataloaders(config_path: str, use_cv: bool = False):
#     """
#     Read config.yaml and return either simple splits or CV dataloaders.
#     """
#     cfg = load_config(config_path)
#     data_cfg = cfg['data']
#     root = data_cfg['root']
#     batch_size = data_cfg['batch_size']
#     num_workers = data_cfg.get('num_workers', 4)
#     pin_memory = data_cfg.get('pin_memory', True)
#     persistent_workers = data_cfg.get('persistent_workers', True)
#     prefetch_factor = data_cfg.get('prefetch_factor', 2)
#     random_state = cfg['training'].get('seed', 19)

#     if not use_cv:
#         return make_split_loaders(
#             dataset_root=root,
#             batch_size=batch_size,
#             random_state=random_state,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             persistent_workers=persistent_workers,
#             prefetch_factor=prefetch_factor,
#             normalization=True
#         )
#     else:
#         return make_cv_loaders(
#             dataset_root=root,
#             batch_size=batch_size,
#             random_state=random_state,
#             num_workers=num_workers,
#             pin_memory=pin_memory,
#             persistent_workers=persistent_workers,
#             prefetch_factor=prefetch_factor,
#             normalization=True
#         )