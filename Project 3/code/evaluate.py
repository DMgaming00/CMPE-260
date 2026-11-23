"""Inference/evaluation demo: reload a saved checkpoint and score it.

    python evaluate.py --checkpoint results/best_model.pt
    python evaluate.py --checkpoint results/best_model.pt --nodes 0 1 2 1700

Prints accuracy and macro-F1 on each split, per-class test accuracy, the test
confusion matrix, and (optionally) predictions for individual node ids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models import build_model
from train import load_dataset, macro_f1, masked_accuracy, resolve_device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='results/best_model.pt')
    p.add_argument('--data-dir', default=None, help='defaults to the training data dir')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    p.add_argument('--nodes', type=int, nargs='*', default=[],
                   help='node ids to show individual predictions for')
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f'Checkpoint {ckpt_path} not found. Run train.py first.')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' not in ckpt:
        raise SystemExit('Checkpoint format not recognised (expected output of this repo\'s train.py).')
    cfg = ckpt['config']

    device = resolve_device(args.device)
    dataset = load_dataset(cfg['dataset'], args.data_dir or cfg['data_dir'])
    data = dataset[0].to(device)
    model = build_model(cfg['model'], ckpt['num_features'], ckpt['num_classes'],
                        hidden=cfg['hidden'], dropout=cfg['dropout'],
                        num_layers=cfg['num_layers'], K=cfg['K'], alpha=cfg['alpha'],
                        heads=cfg['heads']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=-1)
    C = dataset.num_classes

    report = {'model': cfg['model'], 'dataset': cfg['dataset'], 'seed': cfg['seed']}
    for split in ('train', 'val', 'test'):
        mask = data[f'{split}_mask']
        report[f'{split}_accuracy'] = round(masked_accuracy(logits, data.y, mask), 4)
        report[f'{split}_macro_f1'] = round(macro_f1(pred[mask], data.y[mask], C), 4)
    print(json.dumps(report, indent=2))

    tm = data.test_mask
    y_true, y_pred = data.y[tm].cpu(), pred[tm].cpu()
    conf = torch.zeros(C, C, dtype=torch.long)
    for t, q in zip(y_true, y_pred):
        conf[t, q] += 1
    print('\nPer-class test accuracy (class: correct/total):')
    for c in range(C):
        total = int(conf[c].sum())
        print(f'  class {c}: {int(conf[c, c])}/{total} = {conf[c, c] / max(total, 1):.3f}')
    print('\nTest confusion matrix (rows = true class, cols = predicted):')
    print(conf.numpy())

    if args.nodes:
        probs = logits.softmax(dim=-1).cpu()
        print('\nNode predictions:')
        for v in args.nodes:
            if not 0 <= v < data.num_nodes:
                print(f'  node {v}: out of range (0..{data.num_nodes - 1})')
                continue
            split = next((s for s in ('train', 'val', 'test') if bool(data[f'{s}_mask'][v])), 'unlabelled')
            print(f'  node {v:5d} [{split:10s}] true={int(data.y[v])} pred={int(pred[v])} '
                  f'p={probs[v].max():.3f}')


if __name__ == '__main__':
    main()
