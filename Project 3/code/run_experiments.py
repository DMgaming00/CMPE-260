"""Multi-seed experiments reported in the survey (Section XV) and the slides.

    python run_experiments.py                    # full study, ~10-20 min on a laptop CPU
    python run_experiments.py --quick            # 2 seeds, Cora only (~1 min sanity check)

Two studies are run, both on the public Planetoid split:

1. ``baselines``: MLP, GCN, SGC, APPNP and GAT on Cora, CiteSeer and PubMed,
   ``--seeds`` seeds each (default 10: seeds 0..9).
2. ``depth``: plain GCN with 1, 2, 4, 8, 16 and 32 layers on Cora, illustrating the
   depth degradation analysed by Oono and Suzuki [7].

Each run is a fresh call of ``train.train`` with the same selection protocol.
Per-run rows go to results/experiments/runs.csv; mean +- std summaries go to
results/experiments/summary.csv, depth.csv and summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from train import build_parser, train

MODELS = ['mlp', 'gcn', 'sgc', 'appnp', 'gat']
DATASETS = ['Cora', 'CiteSeer', 'PubMed']
DEPTHS = [1, 2, 4, 8, 16, 32]


def run_one(data_dir: str, device: str, **overrides) -> dict:
    args = build_parser().parse_args([])
    args.data_dir, args.device, args.output_dir = data_dir, device, ''
    for k, v in overrides.items():
        setattr(args, k, v)
    return train(args, verbose=False)


def summarise(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return 100 * mean, 100 * std


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='results/experiments')
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--quick', action='store_true', help='2 seeds, Cora only')
    p.add_argument('--skip-depth', action='store_true')
    a = p.parse_args()

    seeds = list(range(2 if a.quick else a.seeds))
    datasets = ['Cora'] if a.quick else DATASETS
    depths = [1, 2, 8] if a.quick else DEPTHS
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for ds in datasets:
        for m in MODELS:
            for s in seeds:
                r = run_one(a.data_dir, a.device, model=m, dataset=ds, seed=s)
                runs.append({'study': 'baselines', 'model': m, 'dataset': ds, 'layers': 2, 'seed': s,
                             'val_acc': r['validation_accuracy'], 'test_acc': r['test_accuracy'],
                             'test_macro_f1': r['test_macro_f1'], 'best_epoch': r['best_epoch'],
                             'train_seconds': r['train_seconds'], 'parameters': r['parameters']})
                print(f"{ds:8s} {m:5s} seed={s} val={r['validation_accuracy']:.3f} "
                      f"test={r['test_accuracy']:.3f} ({r['train_seconds']:.1f}s)", flush=True)
    if not a.skip_depth:
        for L in depths:
            for s in seeds:
                r = run_one(a.data_dir, a.device, model='gcn', dataset='Cora', seed=s, num_layers=L)
                runs.append({'study': 'depth', 'model': 'gcn', 'dataset': 'Cora', 'layers': L, 'seed': s,
                             'val_acc': r['validation_accuracy'], 'test_acc': r['test_accuracy'],
                             'test_macro_f1': r['test_macro_f1'], 'best_epoch': r['best_epoch'],
                             'train_seconds': r['train_seconds'], 'parameters': r['parameters']})
                print(f"depth L={L:2d} seed={s} test={r['test_accuracy']:.3f}", flush=True)

    with (out / 'runs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=runs[0].keys())
        w.writeheader()
        w.writerows(runs)

    def group(study, key_fields):
        groups = {}
        for r in runs:
            if r['study'] == study:
                groups.setdefault(tuple(r[k] for k in key_fields), []).append(r)
        rows = []
        for key, rs in groups.items():
            tm, ts = summarise([r['test_acc'] for r in rs])
            vm, vs = summarise([r['val_acc'] for r in rs])
            fm, fs = summarise([r['test_macro_f1'] for r in rs])
            rows.append({**dict(zip(key_fields, key)), 'n_seeds': len(rs),
                         'test_acc_mean': round(tm, 2), 'test_acc_std': round(ts, 2),
                         'val_acc_mean': round(vm, 2), 'val_acc_std': round(vs, 2),
                         'test_f1_mean': round(fm, 2), 'test_f1_std': round(fs, 2),
                         'mean_train_seconds': round(statistics.fmean(r['train_seconds'] for r in rs), 2),
                         'parameters': rs[0]['parameters']})
        return rows

    summary = group('baselines', ['dataset', 'model'])
    depth = group('depth', ['layers']) if not a.skip_depth else []
    for name, rows in (('summary.csv', summary), ('depth.csv', depth)):
        if rows:
            with (out / name).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
    (out / 'summary.json').write_text(json.dumps({'seeds': seeds, 'baselines': summary,
                                                  'depth': depth}, indent=2) + '\n')
    print('\nTest accuracy, mean +- std over seeds:')
    for r in summary:
        print(f"  {r['dataset']:8s} {r['model']:5s} {r['test_acc_mean']:6.2f} +- {r['test_acc_std']:.2f}")
    for r in depth:
        print(f"  GCN depth {r['layers']:2d}: {r['test_acc_mean']:6.2f} +- {r['test_acc_std']:.2f}")
    print(f'\nwrote {out}/runs.csv, summary.csv, depth.csv, summary.json')


if __name__ == '__main__':
    main()
