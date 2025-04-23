# rgiant/data/dataloader.py

import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from rgiant.data.dataset import ConnectomeDataset
from rgiant.data.transforms import NormalizeNodeFeatures



def compute_train_val_test_idx(y, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Split indices for a single train/val/test split, stratified on y.
    """
    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, stratify=y, random_state=seed
    )
    # relative val size within trainval
    rel_val = val_frac / (1 - test_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=rel_val, stratify=y[idx_trainval], random_state=seed
    )
    return idx_train, idx_val, idx_test



def make_split_loaders(
    data_root: str,
    batch_size: int,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    num_workers: int = 0
):
    """
    Returns three DataLoaders (train, val, test), with each graph normalized
    exactly once on its respective train split.
    """
    # 1) Load all graphs
    full_ds = ConnectomeDataset(root=data_root)

    # 2) Extract labels for stratification
    y = np.array([int(d.y.item()) for d in full_ds])

    # 3) Get indices
    idx_train, idx_val, idx_test = compute_train_val_test_idx(
        y, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    # 4) Compute train‐only mean/std
    all_x = torch.cat([full_ds[i].x for i in idx_train], dim=0)
    mean, std = all_x.mean(0), all_x.std(0)

    # 5) Build normalization transform
    norm = NormalizeNodeFeatures(mean, std)

    # 6) Eagerly normalize each split once
    train_graphs = [ norm(full_ds[i]) for i in idx_train ]
    val_graphs   = [ norm(full_ds[i]) for i in idx_val   ]
    test_graphs  = [ norm(full_ds[i]) for i in idx_test  ]

    # 7) Wrap in DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_graphs,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader



def make_cv_loaders(
    data_root: str,
    batch_size: int,
    n_splits: int = 5,
    seed: int = 42,
    num_workers: int = 0
):
    """
    Yields (fold, train_loader, val_loader) for each fold of a stratified K-fold CV.
    Each graph is normalized exactly once per fold.
    """
    full_ds = ConnectomeDataset(root=data_root)
    y = np.array([int(d.y.item()) for d in full_ds])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        # train‐only stats
        all_x = torch.cat([full_ds[i].x for i in train_idx], dim=0)
        mean, std = all_x.mean(0), all_x.std(0)
        norm = NormalizeNodeFeatures(mean, std)

        # eager normalize
        train_graphs = [ norm(full_ds[i]) for i in train_idx ]
        val_graphs   = [ norm(full_ds[i]) for i in val_idx   ]

        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)
        val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)

        yield fold, train_loader, val_loader
