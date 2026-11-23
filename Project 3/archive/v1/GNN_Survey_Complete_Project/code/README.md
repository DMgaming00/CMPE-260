# Graph Neural Networks: Two-Layer GCN Baseline

This folder contains an original educational PyTorch Geometric implementation of the two-layer GCN baseline described by Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017). It is intended for the node-classification literature survey by Kalhar Patel, Dev Mulchandani, Deep Dhaduk, and Nikita.

## Upstream public baseline reference

https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py

The URL above points to the maintained PyTorch Geometric example; it is **not** a repository for the included team code. To obtain a GitHub URL for the included code, upload this project to a repository owned by the team and replace the placeholder link in the report.

## Requirements

Python 3.10+; pip; internet access on first run to download a public citation dataset. PyTorch Geometric basic GCN functionality can be installed with only PyTorch and torch-geometric. An NVIDIA GPU is optional.

## Setup and run

```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py --dataset Cora --epochs 200 --seed 42
```

If PyTorch installation for your GPU requires a platform-specific wheel, consult https://pytorch.org/get-started/locally/ before installing the other packages.

Additional datasets:

```bash
python train.py --dataset CiteSeer --output-dir results_citeseer
python train.py --dataset PubMed --output-dir results_pubmed
python train.py --dataset Cora --device cpu --output-dir results_cpu
```

## Outputs

The run writes `results/metrics.json`, `results/history.csv`, and `results/best_model.pt`. The script never selects a checkpoint using test accuracy: validation accuracy determines the best epoch, and the test mask is evaluated only after loading the selected checkpoint. The output numbers depend on environment, seed, and dataset; **none are pre-filled or claimed as results in the survey**.

## Method and limitations

`GCNConv` implements normalized neighborhood propagation. The model uses two graph convolution layers, ReLU, dropout, cross-entropy on training-mask nodes, Adam, weight decay, and early stopping by validation accuracy. It processes the full citation graph transductively; masks prevent training on held-out labels, but features and graph edges remain visible. This is not a test of inductive transfer or a proof of statistically significant gains. For multi-seed experiments, use different `--seed` values and aggregate the output metrics independently.

## Troubleshooting

If network access is blocked, the initial Planetoid dataset download will fail; run once with network access or supply the dataset in the expected `data/` directory. If GPU memory is insufficient, use `--device cpu`. If Python cannot import `torch_geometric`, verify that package installation occurred in the active virtual environment. For optional PyG accelerated extensions, consult https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html .

## License / attribution

This code was written for the survey using the publicly documented GCN architecture and PyTorch Geometric APIs. Cite Kipf and Welling (ICLR 2017) for the algorithm and PyTorch Geometric for the framework. It is not a verbatim copy of the upstream example.
