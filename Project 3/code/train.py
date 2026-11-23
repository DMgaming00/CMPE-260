"""Reproducible full-batch training for GNN node classification on Planetoid graphs.

Default behaviour trains the two-layer GCN baseline of Kipf and Welling [1] on
Cora with the public Planetoid split [Yang et al., 2016]:

    python train.py --dataset Cora --seed 42

Protocol (identical for every model):
  * parameters are updated using the loss on training-mask nodes only;
  * the checkpoint is selected by validation accuracy (ties -> lower validation loss);
  * the test mask is evaluated exactly once, after the selected checkpoint is restored.

Outputs (in --output-dir): metrics.json, history.csv, best_model.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from models import DEFAULTS, build_model

DATASETS = ('Cora', 'CiteSeer', 'PubMed')


# --------------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only: some scatter kernels have no deterministic CUDA variant.
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(choice: str) -> torch.device:
    if choice == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if choice == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested but CUDA is not available; use --device cpu')
    return torch.device(choice)


def load_dataset(name: str, data_dir: str):
    """Load a Planetoid citation graph with row-normalised features (as in [1])."""
    if name not in DATASETS:
        raise ValueError(f'dataset must be one of {DATASETS}')
    try:
        return Planetoid(root=str(Path(data_dir) / name), name=name,
                         transform=NormalizeFeatures())
    except Exception as exc:  # network failure, missing scipy, corrupt cache ...
        raise RuntimeError(
            f'Could not load {name}. The first run downloads the Planetoid files from '
            f'GitHub (network needed) and requires scipy. If a previous download was '
            f'interrupted, delete {Path(data_dir) / name} and retry. Original error: {exc!r}'
        ) from exc


def masked_accuracy(logits, labels, mask) -> float:
    if int(mask.sum()) == 0:
        raise ValueError('Evaluation mask is empty')
    return (logits[mask].argmax(dim=-1) == labels[mask]).float().mean().item()


def macro_f1(pred, labels, num_classes: int) -> float:
    """Macro-averaged F1 computed with torch only (no scikit-learn dependency)."""
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (labels == c)).sum().item()
        fp = ((pred == c) & (labels != c)).sum().item()
        fn = ((pred != c) & (labels == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


# ------------------------------------------------------------------------ training
def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Replace unset (None) hyperparameters by the model-specific defaults."""
    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def train(args: argparse.Namespace, verbose: bool = True) -> dict:
    """Train one model with one seed; return the metrics dictionary."""
    args = fill_defaults(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = load_dataset(args.dataset, args.data_dir)
    data = dataset[0].to(device)

    model = build_model(args.model, dataset.num_features, dataset.num_classes,
                        hidden=args.hidden, dropout=args.dropout, num_layers=args.num_layers,
                        K=args.K, alpha=args.alpha, heads=args.heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history, best = [], None
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask]).item()
            train_acc = masked_accuracy(logits, data.y, data.train_mask)
            val_acc = masked_accuracy(logits, data.y, data.val_mask)
        history.append({'epoch': epoch, 'loss': float(loss.item()), 'val_loss': val_loss,
                        'train_accuracy': train_acc, 'val_accuracy': val_acc})

        # Selection uses validation data only: higher accuracy, ties -> lower loss.
        if best is None or (val_acc, -val_loss) > (best['val_acc'], -best['val_loss']):
            best = {'epoch': epoch, 'val_acc': val_acc, 'val_loss': val_loss,
                    'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(f'epoch={epoch:03d} loss={loss.item():.4f} val_loss={val_loss:.4f} '
                  f'train={train_acc:.4f} val={val_acc:.4f}')
        if stale >= args.patience:
            break
    train_seconds = time.perf_counter() - start

    # Restore the validation-selected checkpoint; touch the test mask once.
    model.load_state_dict(best['state'])
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=-1)
        test_acc = masked_accuracy(logits, data.y, data.test_mask)
        test_f1 = macro_f1(pred[data.test_mask], data.y[data.test_mask], dataset.num_classes)

    config = {k: v for k, v in vars(args).items()}
    results = {
        'model': args.model, 'dataset': args.dataset, 'seed': args.seed, 'device': str(device),
        'best_epoch': best['epoch'], 'validation_accuracy': best['val_acc'],
        'test_accuracy': test_acc, 'test_macro_f1': test_f1,
        'epochs_run': len(history), 'train_seconds': round(train_seconds, 3),
        'num_nodes': int(data.num_nodes), 'num_directed_edges': int(data.num_edges),
        'parameters': sum(p.numel() for p in model.parameters()),
        'protocol': 'Planetoid public split; checkpoint chosen on validation; test evaluated once',
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'torch_geometric': torch_geometric.__version__},
        'config': config,
    }

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        (out_dir / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
        torch.save({'state_dict': best['state'], 'config': config,
                    'num_features': dataset.num_features, 'num_classes': dataset.num_classes},
                   out_dir / 'best_model.pt')
    if verbose:
        print(json.dumps({k: v for k, v in results.items() if k != 'config'}, indent=2))
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=sorted(DEFAULTS), default='gcn',
                   help='gcn (baseline) or a control: mlp, sgc, appnp, gat')
    p.add_argument('--dataset', choices=DATASETS, default='Cora')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-dir', default='results',
                   help="directory for metrics/history/checkpoint ('' disables writing)")
    # None -> use the model-specific default from models.DEFAULTS
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--hidden', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight-decay', type=float, default=None)
    p.add_argument('--num-layers', type=int, default=2, help='GCN depth (depth experiment)')
    p.add_argument('--K', type=int, default=None, help='propagation steps (SGC: 2, APPNP: 10)')
    p.add_argument('--alpha', type=float, default=0.1, help='APPNP teleport probability')
    p.add_argument('--heads', type=int, default=8, help='GAT attention heads')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--log-every', type=int, default=20)
    return p


if __name__ == '__main__':
    train(build_parser().parse_args())
