#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import trange
from sklearn.metrics import classification_report

# Own packages
from rgiant.data.dataloader import make_split_loaders
from rgiant.models.gnn_baseline import GNNClassifier
from rgiant.training.utils import set_seed, EarlyStopping, LearningCurvePlotter


# ──────────────────────────────────────────────────────────────
# Model factory: map a name to a class
# ──────────────────────────────────────────────────────────────
MODEL_FACTORY = {
    'baseline': GNNClassifier,
}

# ──────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(description="Modular GNN Training Script")
    # Model / data
    p.add_argument('--model',           choices=MODEL_FACTORY, required=True)
    p.add_argument('--data-root',       type=str,  required=True)
    p.add_argument('--batch-size',      type=int,  default=16)
    p.add_argument('--use-sampler',     action='store_true')
    p.add_argument('--use-wloss',       action='store_true')
    # Training hyperparams  
    p.add_argument('--epochs',          type=int,  default=100)
    p.add_argument('--lr',              type=float,default=1e-4)
    p.add_argument('--patience',        type=int,  default=10)
    p.add_argument('--sampler-power',   type=float,default=1.0)
    # Misc  
    p.add_argument('--seed',            type=int,  default=19)
    p.add_argument('--device',          type=str,  default='cuda')
    p.add_argument('--plots-root',      type=str,  default='plots/')
    p.add_argument('-v', '--verbose',   action='store_true')
    return p.parse_args()

# ──────────────────────────────────────────────────────────────
# Training and evaluation loops (unchanged)
# ──────────────────────────────────────────────────────────────
def train_step(model, loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0
    correct = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, batch=data.batch)
        loss = loss_fn(out, data.y if out.dim()>1 else data.y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1) if out.dim()>1 else (torch.sigmoid(out)>0.5).long()
        correct   += (pred == data.y).sum().item()

    avg_loss = total_loss/len(loader.dataset)
    accuracy = correct/len(loader.dataset)

    return avg_loss, accuracy

@torch.no_grad()
def eval_step(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0
    correct = 0
    y_pred = []
    y_true = []

    for data in loader:
        data = data.to(device)

        out = model(data.x, data.edge_index, batch=data.batch)  # [B, num_classes]
        loss = loss_fn(out, data.y)                             # Mean loss per graph over the epoch
        total_loss += loss.item() * data.num_graphs             # We multiply by num graphs in batch to recover total loss again

        pred = out.argmax(dim=1)                                # [B]
        correct += (pred == data.y).sum().item()                # sum all bools at indices where prediction was correct

        y_pred.append(pred.cpu())
        y_true.append(data.y.cpu())

    y_pred = torch.cat(y_pred).numpy()  # [B * num_batches] 1D
    y_true = torch.cat(y_true).numpy()

    # overall metrics
    avg_loss = total_loss / len(loader.dataset)
    acc      = correct    / len(loader.dataset)

    # per-class arrays
    per_p = precision_score(y_true, y_pred, average=None, zero_division=0)  # [num_classes]
    per_r = recall_score   (y_true, y_pred, average=None, zero_division=0)
    per_f = f1_score       (y_true, y_pred, average=None, zero_division=0)

    # macro & weighted
    mac_p = precision_score(y_true, y_pred, average="macro",   zero_division=0)     # scalar
    mac_r = recall_score   (y_true, y_pred, average="macro",   zero_division=0)
    mac_f = f1_score       (y_true, y_pred, average="macro",   zero_division=0)

    wtd_p = precision_score(y_true, y_pred, average="weighted",zero_division=0)     # scalar
    wtd_r = recall_score   (y_true, y_pred, average="weighted",zero_division=0)
    wtd_f = f1_score       (y_true, y_pred, average="weighted",zero_division=0)

    return {
        'loss':    avg_loss, 'acc': acc,
        'per_p':   per_p,    'per_r': per_r,    'per_f': per_f,
        'mac_p':   mac_p,    'mac_r': mac_r,    'mac_f': mac_f,
        'wtd_p':   wtd_p,    'wtd_r': wtd_r,    'wtd_f': wtd_f,
        'y_true':  y_true,   'y_pred': y_pred
    }

# ──────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────
def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


    # 1) Reproducibility
    set_seed(args.seed)


    # 2) Data loaders
    loaders = make_split_loaders(
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        use_sampler=args.use_sampler,
        sampler_power=args.sampler_power
    )
    train_loader    = loaders['train_loader']
    val_loader      = loaders['val_loader']
    test_loader     = loaders['test_loader']
    class_weights   = loaders['class_weights'].to(device)


    # 3) Model
    num_classes = len(class_weights)
    num_node_features = train_loader.dataset[0].num_node_features

    ModelClass = MODEL_FACTORY[args.model]
    model = ModelClass(
        in_node_feats=num_node_features,
        num_classes=num_classes
    ).to(device)


    # 4) Loss + optimizer
    if args.use_wloss:
        loss_fn = CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=args.lr)


    # 5) Early stopping & plotting
    # build a dynamic list of metrics to track
    metrics = [
        'train_loss', 'val_loss',
        'train_acc',  'val_acc',
    ]

    # for each class, add per‐class precision, recall & f1 (only val side here)
    for i in range(num_classes):
        metrics += [
            f"val_per_p_{i}",
            f"val_per_r_{i}",
            f"val_per_f_{i}",
        ]

    # then add the aggregated metrics
    metrics += [
        f"val_macro_p",
        f"val_macro_r",
        f"val_macro_f",
    ]

    plotter = LearningCurvePlotter(metrics_to_track=metrics)
    early_stopper = EarlyStopping(patience=args.patience, mode='max')

    # 6) Training loop
    # Wrap epoch loop to handle verbose OR tqdm process bar
    if args.verbose:
        epoch_iterator = range(1, args.epochs + 1)
    else:
        # trange is just `tqdm(range(...))`
        epoch_iterator = trange(1, args.epochs + 1, desc="Epochs")
    
    # Loop over all epochs
    for epoch in epoch_iterator:
        train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, device)
        val = eval_step(model, val_loader, loss_fn, device)
        # unpack a few for convenience
        val_loss, val_acc = val['loss'], val['acc']

        if args.verbose:
            print(f"Epoch {epoch}/{args.epochs} "
                f"- Train: L {train_loss:.4f}, A {train_acc:.4f} "
                f"- Val: L {val_loss:.4f}, A {val_acc:.4f}")#, P {val_prec:.4f}, R {val_rec:.4f}, F1 {val_f1:.4f}")
        
        log_kwargs = {
        'train_loss': train_loss,
        'train_acc':  train_acc,
        'val_loss':   val_loss,
        'val_acc':    val_acc,
        }

        # per‐class precision/recall/f1
        for i in range(num_classes):
            log_kwargs[f"val_per_p_{i}"] = val['per_p'][i]
            log_kwargs[f"val_per_r_{i}"] = val['per_r'][i]
            log_kwargs[f"val_per_f_{i}"] = val['per_f'][i]

        # macro‐averaged
        log_kwargs.update({
            'val_macro_p': val['mac_p'],
            'val_macro_r': val['mac_r'],
            'val_macro_f': val['mac_f'],
        })
        # # weighted‐averaged
        # log_kwargs.update({
        #     'val_wtd_p': val['wtd_p'],
        #     'val_wtd_r': val['wtd_r'],
        #     'val_wtd_f': val['wtd_f'],
        # })

        # 5) Log everything
        plotter.log(**log_kwargs)

        early_stopper(val['mac_f'], model)
        if early_stopper.early_stop:
            break

    if early_stopper.early_stop: print(f"Early stopping at epoch {epoch}")
    
    # 7) Save & plot
    os.makedirs(args.plots_root, exist_ok=True)

    # Build a filename that encodes all CLI arguments + model specs
    parts = [
        args.model,
        f"bs{args.batch_size}",
        f"sampler{int(args.use_sampler)}",
        f"wloss{int(args.use_wloss)}",
        f"epochs{args.epochs}",
        f"lr{args.lr:.0e}",               # scientific notation
        f"pat{args.patience}",
        f"sp{args.sampler_power}",
        f"seed{args.seed}",
        f"cls{len(class_weights)}"
    ]
    filename = "_".join(parts) + ".png"

    save_path = os.path.join(args.plots_root, filename)
    plotter.plot(save_path=save_path)

    # # 8) Final test eval
    # test_loss, test_acc, test_prec, test_rec, test_f1, test_y, test_y_pred = eval_step(model, test_loader, loss_fn, device)
    # # print(f"\nTest: L {test_loss:.4f}, A {test_acc:.4f}, P {test_prec:.4f}, R {test_rec:.4f}, F1 {test_f1:.4f}")
    # print(classification_report(
    #     test_y, test_y_pred,
    #     target_names=['HC','MCI & AD'],
    #     digits=3,
    #     zero_division=0
    #     ))

if __name__ == '__main__':
    main()
