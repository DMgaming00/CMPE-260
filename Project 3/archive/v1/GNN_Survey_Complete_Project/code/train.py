"""Reproducible two-layer GCN baseline for Planetoid citation graphs."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures


class GCN(nn.Module):
    """Kipf--Welling-style, full-batch, two-layer graph convolutional network."""

    def __init__(self, num_features, hidden, num_classes, dropout):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def accuracy(logits, labels, mask):
    """Return accuracy on the indicated nodes only."""
    if mask.sum().item() == 0:
        raise ValueError('Evaluation mask is empty')
    return (logits[mask].argmax(dim=-1) == labels[mask]).float().mean().item()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args):
    set_seed(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    dataset = Planetoid(root=str(Path(args.data_dir) / args.dataset),
                       name=args.dataset, transform=NormalizeFeatures())
    data = dataset[0].to(device)
    model = GCN(dataset.num_features, args.hidden, dataset.num_classes, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = -1.0
    best_epoch = 0
    stale = 0
    best_state = None
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
            train_acc = accuracy(logits, data.y, data.train_mask)
            val_acc = accuracy(logits, data.y, data.val_mask)
        history.append({'epoch': epoch, 'loss': float(loss.item()),
                        'train_accuracy': train_acc, 'val_accuracy': val_acc})
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(f'epoch={epoch:03d} loss={loss.item():.4f} '
                  f'train={train_acc:.4f} val={val_acc:.4f}')
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_acc = accuracy(logits, data.y, data.test_mask)
    results = {'dataset': args.dataset, 'seed': args.seed, 'device': str(device),
               'best_epoch': best_epoch, 'validation_accuracy': best_val,
               'test_accuracy': test_acc, 'epochs_run': len(history),
               'num_nodes': int(data.num_nodes), 'num_edges': int(data.num_edges),
               'parameters': sum(p.numel() for p in model.parameters()),
               'protocol': 'Planetoid public masks; test used once after validation selection'}
    with (out_dir / 'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (out_dir / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    torch.save(best_state, out_dir / 'best_model.pt')
    print(json.dumps(results, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['Cora', 'CiteSeer', 'PubMed'], default='Cora')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output-dir', default='results')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--log-every', type=int, default=20)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
