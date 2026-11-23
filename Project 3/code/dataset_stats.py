"""Print descriptive statistics of the Planetoid graphs used in the experiments.

    python dataset_stats.py            # writes results/dataset_stats.json

Edge homophily is the fraction of edges whose endpoints share a label
(Zhu et al., NeurIPS 2020). It uses ALL labels and is therefore a dataset
diagnostic only -- it is never fed to a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch_geometric.utils import contains_self_loops, homophily, is_undirected

from train import DATASETS, load_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--output', default='results/dataset_stats.json')
    args = p.parse_args()

    rows = []
    for name in DATASETS:
        data = load_dataset(name, args.data_dir)[0]
        rows.append({
            'dataset': name,
            'nodes': int(data.num_nodes),
            'undirected_edges': int(data.num_edges) // 2,
            'features': int(data.num_features),
            'classes': int(data.y.max()) + 1,
            'train': int(data.train_mask.sum()),
            'val': int(data.val_mask.sum()),
            'test': int(data.test_mask.sum()),
            'label_rate': round(int(data.train_mask.sum()) / int(data.num_nodes), 4),
            'edge_homophily': round(float(homophily(data.edge_index, data.y, method='edge')), 3),
            'undirected': bool(is_undirected(data.edge_index)),
            'self_loops': bool(contains_self_loops(data.edge_index)),
        })
    header = f"{'dataset':9s} {'nodes':>6s} {'edges':>6s} {'feat':>5s} {'cls':>3s} " \
             f"{'train':>5s} {'val':>4s} {'test':>5s} {'h_edge':>6s}"
    print(header)
    for r in rows:
        print(f"{r['dataset']:9s} {r['nodes']:6d} {r['undirected_edges']:6d} {r['features']:5d} "
              f"{r['classes']:3d} {r['train']:5d} {r['val']:4d} {r['test']:5d} {r['edge_homophily']:6.3f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + '\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
