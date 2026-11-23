"""Regenerate every figure used in the report and the slides.

    python make_figures.py --results results/experiments --history results/history.csv

Data figures are drawn ONLY from files written by run_experiments.py and
train.py; nothing is typed in by hand. Conceptual diagrams (message passing,
taxonomy) need no data. Output: ../figures/*.png (repository-level folder).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Categorical palette validated for colour-vision deficiency (fixed slot order).
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
MODEL_ORDER = ['mlp', 'sgc', 'gcn', 'appnp', 'gat']
MODEL_LABEL = {'mlp': 'MLP (no graph)', 'sgc': 'SGC [5]', 'gcn': 'GCN [1]',
               'appnp': 'APPNP [6]', 'gat': 'GAT [3]'}
INK, MUTED, GRID = '#1f2328', '#5b616b', '#e4e6ea'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200,
})


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------- data plots
def fig_baselines(summary: list[dict], out: Path) -> None:
    datasets = ['Cora', 'CiteSeer', 'PubMed']
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    width = 0.16
    for i, m in enumerate(MODEL_ORDER):
        rows = {r['dataset']: r for r in summary if r['model'] == m}
        xs = [d + (i - 2) * width for d in range(len(datasets))]
        means = [float(rows[d]['test_acc_mean']) for d in datasets]
        stds = [float(rows[d]['test_acc_std']) for d in datasets]
        ax.bar(xs, means, width * 0.88, yerr=stds, color=PALETTE[i], label=MODEL_LABEL[m],
               error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2), zorder=3)
        for x, v in zip(xs, means):
            ax.text(x, v + 1.6, f'{v:.1f}', ha='center', va='bottom', fontsize=6.5, color=INK)
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(40, 92)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    n = summary[0]['n_seeds']
    ax.set_title(f'Planetoid public split, mean ± std over {n} seeds (this project)',
                 fontsize=10, color=INK, loc='left')
    ax.legend(ncol=5, fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(out / 'fig_baselines.png', bbox_inches='tight')
    plt.close(fig)


def fig_depth(depth: list[dict], out: Path) -> None:
    depth = sorted(depth, key=lambda r: int(r['layers']))
    L = [int(r['layers']) for r in depth]
    m = [float(r['test_acc_mean']) for r in depth]
    s = [float(r['test_acc_std']) for r in depth]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.fill_between(L, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                    color=PALETTE[0], alpha=0.15, lw=0)
    ax.plot(L, m, color=PALETTE[0], lw=2, marker='o', ms=6, zorder=3)
    for x, y in zip(L, m):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, color=INK)
    ax.set_xscale('log', base=2)
    ax.set_xticks(L, [str(x) for x in L])
    ax.set_xlabel('Number of GCN layers (no residuals)')
    ax.set_ylabel('Cora test accuracy (%)')
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID)
    ax.set_title(f'Plain GCN depth sweep, mean ± std over {depth[0]["n_seeds"]} seeds',
                 fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    fig.savefig(out / 'fig_depth.png', bbox_inches='tight')
    plt.close(fig)


def fig_curves(history: list[dict], out: Path) -> None:
    ep = [int(r['epoch']) for r in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    a1.plot(ep, [float(r['loss']) for r in history], color=PALETTE[0], lw=2, label='train loss')
    a1.plot(ep, [float(r['val_loss']) for r in history], color=PALETTE[1], lw=2, label='validation loss')
    a1.set_xlabel('Epoch'); a1.set_ylabel('Cross-entropy'); a1.legend(frameon=False, fontsize=8)
    a2.plot(ep, [100 * float(r['train_accuracy']) for r in history], color=PALETTE[0], lw=2, label='train')
    a2.plot(ep, [100 * float(r['val_accuracy']) for r in history], color=PALETTE[1], lw=2, label='validation')
    best = max(history, key=lambda r: (float(r['val_accuracy']), -float(r['val_loss'])))
    a2.axvline(int(best['epoch']), color=MUTED, lw=1, ls='--')
    a2.annotate(f"selected\nepoch {best['epoch']}", (int(best['epoch']), 55), xytext=(-6, 0),
                textcoords='offset points', ha='right', fontsize=8, color=MUTED)
    a2.set_xlabel('Epoch'); a2.set_ylabel('Accuracy (%)'); a2.legend(frameon=False, fontsize=8, loc='lower right')
    for a in (a1, a2):
        a.yaxis.grid(True, color=GRID)
    fig.suptitle('Two-layer GCN on Cora (seed 42): training dynamics', x=0.01, ha='left',
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / 'fig_training_curves.png', bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------- conceptual diagrams
def _box(ax, x, y, w, h, text, fc, ec=None, fs=9, color=INK, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06',
                                fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color,
            weight=weight, wrap=True)


def _arrow(ax, p, q, color=MUTED):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, color=color, lw=1.2))


def fig_message_passing(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.6); ax.axis('off')
    # small graph: target v with 3 neighbours
    pos = {'v': (1.6, 1.8), 'a': (0.4, 3.0), 'b': (0.3, 0.7), 'c': (2.9, 3.1), 'd': (2.9, 0.5)}
    for u in 'abcd':
        ax.plot(*zip(pos['v'], pos[u]), color=MUTED, lw=1.2, zorder=1)
    for k, (x, y) in pos.items():
        ax.scatter([x], [y], s=520 if k == 'v' else 360, color=PALETTE[1] if k == 'v' else PALETTE[0],
                   zorder=2, edgecolors='white', linewidths=2)
        ax.text(x, y, k, ha='center', va='center', color='white', fontsize=10, weight='bold', zorder=3)
    ax.text(1.6, 0.0, 'neighbourhood N(v) = {a, b, c, d}', ha='center', fontsize=8, color=MUTED)
    steps = [('MESSAGE', 'm_uv = W h_u\n(+ weight or attention)'),
             ('AGGREGATE', 'sum / mean / max\nor α-weighted'),
             ('UPDATE', 'h_v′ = σ(agg, h_v)\n(self / skip / MLP)')]
    for i, (t, d) in enumerate(steps):
        x = 3.8 + i * 2.1
        _box(ax, x, 1.15, 1.8, 1.5, '', '#eef3fb', '#c9d8f0')
        ax.text(x + 0.9, 2.3, t, ha='center', fontsize=9, weight='bold', color=INK)
        ax.text(x + 0.9, 1.65, d, ha='center', va='center', fontsize=7.8, color=MUTED)
        _arrow(ax, (x - 0.35 if i else 3.15, 1.9), (x - 0.02, 1.9))
    ax.text(3.8, 0.45, 'GCN: fixed weights 1/√(d̃_u d̃_v)   ·   GAT: learned α_uv   ·   '
            'GraphSAGE: sampled N(v)   ·   GIN: sum + MLP', fontsize=7.5, color=INK)
    fig.savefig(out / 'fig_message_passing.png', bbox_inches='tight')
    plt.close(fig)


def fig_taxonomy(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')
    _box(ax, 2.9, 4.2, 4.2, 0.6, 'GNNs for node classification', '#1f2a44', fs=10,
         color='white', weight='bold')
    cols = [
        ('What is aggregated\nand how', ['GCN [1] fixed', 'GAT [3] attention', 'GIN [4] injective sum']),
        ('Where propagation\nhappens', ['SGC [5] pre-compute', 'APPNP [6] post-predict']),
        ('Limits of depth\n& expressivity', ['Oono–Suzuki [7]', 'WL bound [4]']),
        ('Scale &\ndeployment', ['GraphSAGE [2]', 'GraphSAINT [8]']),
        ('Data & evaluation\nassumptions', ['Geom-GCN [10]', 'OGB [9]']),
    ]
    w, gap = 1.8, 0.2
    for i, (head, items) in enumerate(cols):
        x = 0.1 + i * (w + gap)
        ax.plot([5, x + w / 2], [4.2, 3.55], color=MUTED, lw=1)
        _box(ax, x, 2.95, w, 0.6, head, PALETTE[i], fs=7.4, color='white', weight='bold')
        for j, it in enumerate(items):
            _box(ax, x + 0.08, 2.15 - j * 0.68, w - 0.16, 0.52, it, '#f2f4f7', '#d5d9e0', fs=7.0)
    fig.savefig(out / 'fig_taxonomy.png', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='results/experiments')
    p.add_argument('--history', default='results/history.csv')
    p.add_argument('--out', default='../figures')
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_message_passing(out)
    fig_taxonomy(out)
    res = Path(a.results)
    if (res / 'summary.csv').exists():
        fig_baselines(read_csv(res / 'summary.csv'), out)
    else:
        print(f'skip fig_baselines: {res / "summary.csv"} missing (run run_experiments.py)')
    if (res / 'depth.csv').exists():
        fig_depth(read_csv(res / 'depth.csv'), out)
    else:
        print(f'skip fig_depth: {res / "depth.csv"} missing')
    if Path(a.history).exists():
        fig_curves(read_csv(Path(a.history)), out)
    else:
        print(f'skip fig_training_curves: {a.history} missing (run train.py)')
    print(f'figures written to {out.resolve()}')


if __name__ == '__main__':
    main()
