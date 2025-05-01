import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from rgiant.models.gnn_baseline import GNNClassifier
from rgiant.data.dataloader import make_split_loaders  # Your pre-split train/val/test loader
from sklearn.metrics import precision_score, recall_score, f1_score
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
            print(f"No improvement for {self.counter} epoch(s)")

        if self.counter >= self.patience:
            self.early_stop = True



class LearningCurvePlotter:
    def __init__(self, metrics_to_track=None):
        """
        metrics_to_track: list of str, e.g.
          ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_precision', 'val_precision']
        """
        self.history = {}
        self.metrics = metrics_to_track or []
        for metric in self.metrics:
            self.history[metric] = []

    def log(self, **kwargs):
        """
        Log values via keyword args. Only metrics in metrics_to_track are recorded.
        E.g. plotter.log(train_loss=..., val_loss=..., val_acc=...)
        """
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)

    def plot(self, save_path=None, max_cols=2):
        """
        Plot grouped train/val curves for each base metric.
        - save_path: if provided, saves figure to this path instead of showing.
        - max_cols: maximum subplots per row.
        """
        # Determine base metrics (everything after 'train_' or 'val_')
        bases = []
        for m in self.metrics:
            if '_' in m:
                base = m.split('_', 1)[1]
                if base not in bases:
                    bases.append(base)
            else:
                bases.append(m)

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
            for prefix in ('train', 'val'):
                key = f"{prefix}_{base}"
                if key in self.history and self.history[key]:
                    ax.plot(
                        range(1, len(self.history[key]) + 1),
                        self.history[key],
                        label=prefix
                    )
            ax.set_title(base.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel(base.replace('_', ' ').title())
            ax.legend()
            ax.grid(True)

        # hide any unused subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 150
LR = 1e-4
DATA_ROOT       = "data/deeplearning_datasets/graphs no-PET tri-label/"
# RAW_FILENAME    = "raw_pet_tri_graphs.pt"
# NORM_FILENAME   = "normalised_pet_tri_graphs.pt"

# ──────────────────────────────────────────────────────────────
# Load preprocessed data
# ──────────────────────────────────────────────────────────────

loaders = make_split_loaders(
    dataset_root=DATA_ROOT, 
    batch_size=BATCH_SIZE
    )

train_loader = loaders["train_loader"]
val_loader = loaders["val_loader"]
test_loader = loaders["test_loader"]
class_weights = loaders["class_weights"]

# ──────────────────────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────────────────────
#model = GNNClassifier(in_node_feats=11).to(DEVICE)
model = GNNClassifier(in_node_feats=10, num_classes=3).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR)
loss_fn = CrossEntropyLoss(weight=class_weights.to(DEVICE))

# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────
def train(loader):
    model.train()
    total_loss = 0
    correct = 0

    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, batch=data.batch)

        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

        all_preds.append(pred.cpu())
        all_labels.append(data.y.cpu())

    # Flatten all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute metrics (macro = unweighted mean across classes)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc, precision, recall, f1


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0

    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, batch=data.batch)
        loss = loss_fn(out, data.y)
        total_loss += loss.item() * data.num_graphs

        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

        all_preds.append(pred.cpu())
        all_labels.append(data.y.cpu())

    # Flatten all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute metrics (macro = unweighted mean across classes)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    acc = correct / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, acc, precision, recall, f1

# ──────────────────────────────────────────────────────────────
# Run training
# ──────────────────────────────────────────────────────────────
early_stopper = EarlyStopping(patience=20, mode="max")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train()
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(val_loader)

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, f1: {val_f1:.4f}")
    
    early_stopper(val_f1, model)

    if early_stopper.early_stop:
        print(f"\n Early stopping at epoch {epoch}")
        model.load_state_dict(early_stopper.best_state)
        break


# Test at the end
test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(test_loader)
print(f"\nTest Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, f1: {test_f1:.4f}")
