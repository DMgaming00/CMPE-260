# Code: GCN baseline and controls

The full setup, usage, expected outputs and troubleshooting are in the [repository README](../README.md). Quick start, run from this folder:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q tests                             # offline smoke tests
python train.py --dataset Cora --seed 42              # two-layer GCN baseline
python evaluate.py --checkpoint results/best_model.pt # reload + evaluate
```

| File | Purpose |
|---|---|
| `models.py` | `GCN` (baseline, Kipf & Welling 2017) and the controls `MLP`, `SGC`, `APPNPNet`, `GAT`; default hyperparameters in `DEFAULTS` |
| `train.py` | Data loading (Planetoid + row-normalized features), full-batch training, validation-based checkpoint selection, a single test evaluation, and outputs (`metrics.json`, `history.csv`, `best_model.pt`) |
| `evaluate.py` | Reloads a checkpoint and prints per-split accuracy and macro-F1, the confusion matrix, and per-node predictions |
| `run_experiments.py` | All models × Cora/CiteSeer/PubMed × 10 seeds, plus the GCN depth sweep |
| `dataset_stats.py` | Node, edge and split counts and edge homophily |
| `make_figures.py` | Figures for the report (reads only files written by the scripts above) |
| `tests/test_smoke.py` | Offline tests on a synthetic graph: output shapes, loss decrease, seeding, metrics |

The upstream PyTorch Geometric GCN example (<https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py>) was used as an API reference only; this code was written for the project.
