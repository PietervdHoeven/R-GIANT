import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt



class EarlyStopping:
    def __init__(self, patience=5, mode='min', delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            mode (str): 'min' for loss, 'max' for incrementing metrics such as accuracy/f1-score.
            delta (float): Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        improved = False

        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
            improved = True
        else:
            if self.mode == 'min':
                if score < self.best_score - self.delta:
                    improved = True
            elif self.mode == 'max':
                if score > self.best_score + self.delta:
                    improved = True

        if improved:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            #print(f"No improvement for {self.counter} epoch(s)")

        if self.counter >= self.patience:
            self.early_stop = True



class LearningCurvePlotter:
    def __init__(self, metrics_to_track=None):
        """
        metrics_to_track: list of str, e.g.
        ['train_loss',      'val_loss',         'train_acc',        'val_acc',
        'val_per_p_{i}',    'val_per_r_{i}',    'val_per_f_{i}',
        'val_mac_p',        'val_mac_r',        'val_mac_f',
        'val_wtd_p',        'val_wtd_r',        'val_wtd_f']
        """
        self.history = {}   # Init empty dict to store histories per metric
        self.metrics = metrics_to_track  # Save metrics to track
        for metric in self.metrics:
            self.history[metric] = []   # init empty list per metric to store historical values in

    def log(self, **kwargs):
        """
        Log values via keyword args. Only metrics in metrics_to_track are recorded.
        E.g. plotter.log(train_loss=..., val_loss=..., val_acc=...)
        """
        for key, value in kwargs.items():   # update all the histories of each metric in kwargs
            if key in self.history:
                self.history[key].append(value)

    def plot(self, save_path=None, max_cols=2):
        """
        Plot grouped train/val curves for each base metric.
        - save_path: if provided, saves figure to this path instead of showing.
        - max_cols: maximum subplots per row.
        """
        print(self.metrics)
        # identify distinct metric‐types (everything after prefix_)
        bases = list({(parts[1].split('_')[1] if '_' in parts[1] else parts[1])
                        for parts in (m.split('_', 1) for m in self.metrics)
                        })
        prio = ['loss', 'acc']
        head = [b for b in prio if b in bases]
        tail = [b for b in bases if b not in prio]
        bases = head + tail
        print(bases)

        n_plots = len(bases)
        if n_plots == 0:
            print("No metrics to plot.")
            return

        ncols = min(max_cols, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        # flatten axes for easy indexing
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, base in enumerate(bases):
            ax = axes[idx]
            for key in self.metrics:
                if base in key.split('_'):
                    ax.plot(
                        range(1, len(self.history[key])+1),
                        self.history[key],
                        label=key
                    )

            ax.set_title(base.replace('_',' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(base.title())
            ax.legend()
            ax.grid(True)
            if base != 'loss':
                ax.set_ylim(0,1)

        # hide any unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()



def set_seed(seed: int = 42):
    """
    Fixes random seeds for reproducibility across Python, NumPy, and PyTorch.
    Call this once at the start of your script (before any randomness).
    """
    # 1) Python built-in
    random.seed(seed)
    # 2) NumPy
    np.random.seed(seed)
    # 3) PyTorch (CPU)
    torch.manual_seed(seed)
    # 4) PyTorch (all GPUs)
    torch.cuda.manual_seed_all(seed)
    # # 5) CuDNN deterministic settings
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # 6) Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)