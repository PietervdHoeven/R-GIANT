#!/usr/bin/env python
import os
import argparse
import torch
import random
import numpy as np
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

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
# Loss factory
# ──────────────────────────────────────────────────────────────
LOSS_FACTORY = {
    'ce':   CrossEntropyLoss,
    'bce':  BCEWithLogitsLoss,
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
    p.add_argument('--num-classes',     type=int,  default=2)
    # Training hyperparams  
    p.add_argument('--epochs',          type=int,  default=100)
    p.add_argument('--lr',              type=float,default=1e-4)
    p.add_argument('--loss',            choices=LOSS_FACTORY, default='ce')
    p.add_argument('--patience',        type=int,  default=10)
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
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)

        out = model(data.x, data.edge_index, batch=data.batch)
        loss = loss_fn(out, data.y if out.dim()>1 else data.y.float())
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1) if out.dim()>1 else (torch.sigmoid(out)>0.5).long()

        correct   += (pred == data.y).sum().item()
        all_preds.append(pred.cpu()); all_labels.append(data.y.cpu())

    all_preds   = torch.cat(all_preds)
    all_labels  = torch.cat(all_labels)

    avg_loss    = total_loss/len(loader.dataset)
    accuracy    = correct/len(loader.dataset)
    precision   = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall      = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1          = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return (avg_loss, accuracy, precision, recall, f1)

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
        batch_size=args.batch_size
    )
    train_loader    = loaders['train_loader']
    val_loader      = loaders['val_loader']
    test_loader     = loaders['test_loader']
    class_weights   = loaders['class_weights'].to(device)

    # 3) Model
    ModelClass = MODEL_FACTORY[args.model]
    model = ModelClass(
        in_node_feats=train_loader.dataset[0].num_node_features,
        num_classes=args.num_classes
    ).to(device)

    # 4) Loss + optimizer
    LossClass = LOSS_FACTORY[args.loss]
    if args.loss == 'ce':
        loss_fn = LossClass(weight=class_weights)
    else:
        # assume BCEWithLogitsLoss; pos_weight = w_pos/w_neg
        pos_weight = class_weights[1]/class_weights[0]
        loss_fn = LossClass(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # 5) Early stopping & plotting
    early_stopper = EarlyStopping(patience=args.patience, mode='max')
    plotter = LearningCurvePlotter(metrics_to_track=[
        'train_loss','train_acc','val_loss','val_acc',
        'val_precision','val_recall','val_f1'
    ])

    # 6) Training loop
    for epoch in tqdm(range(1, args.epochs+1)):
        tr_loss, tr_acc = train_step(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_acc, va_prec, va_rec, va_f1 = eval_step(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch}/{args.epochs} "
              f"- Train: L {tr_loss:.4f}, A {tr_acc:.4f} "
              f"- Val: L {va_loss:.4f}, A {va_acc:.4f}, P {va_prec:.4f}, R {va_rec:.4f}, F1 {va_f1:.4f}")

        plotter.log(
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=va_loss,   val_acc=va_acc,
            val_precision=va_prec, val_recall=va_rec, val_f1=va_f1
        )

        early_stopper(va_f1, model)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # 7) Save & plot
    os.makedirs(args.plots_root, exist_ok=True)
    plotter.plot(save_path=os.path.join(args.plots_root, f"{args.model}_{args.num_classes}classes_{train_loader.dataset[0].num_node_features}features.png"))

    # 8) Final test eval
    te_loss, te_acc, te_prec, te_rec, te_f1 = eval_step(model, test_loader, loss_fn, device)
    print(f"\nTest: L {te_loss:.4f}, A {te_acc:.4f}, P {te_prec:.4f}, R {te_rec:.4f}, F1 {te_f1:.4f}")

if __name__ == '__main__':
    main()
