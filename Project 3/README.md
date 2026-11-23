# Graph Neural Networks for Semi-Supervised Node Classification

**CMPE 260 literature survey and tutorial**, by Kalhar Patel, Dev Mulchandani, Deep Dhaduk and Nikita Memane, San Jose State University

Repository: <https://github.com/DMgaming00/CMPE-260/tree/main/Project%203>

This project surveys ten peer-reviewed papers on graph neural networks (GNNs) for semi-supervised node classification. The papers are GCN, GraphSAGE, GAT, GIN, SGC, APPNP, over-smoothing theory, GraphSAINT, the Open Graph Benchmark, and Geom-GCN. The project also provides a reproducible PyTorch Geometric implementation of the baseline algorithm, the two-layer **Graph Convolutional Network** of Kipf & Welling (ICLR 2017).

| Deliverable | Location |
|---|---|
| Survey paper, IEEE two-column (IEEEtran LaTeX, ~9,000 words, 10 surveyed papers + 6 supporting refs) | [`report/ieee/GNN_Survey_IEEE.tex`](report/ieee/GNN_Survey_IEEE.tex) (compile on Overleaf or locally, see below) |
| Same report as an editable Word document | [`report/GNN_Literature_Survey.docx`](report/GNN_Literature_Survey.docx) (source of both: [`report/survey.md`](report/survey.md)) |
| Lecture slides (62 slides, editable PowerPoint) | [`presentation/GNN_Lecture_60_Slides.pptx`](presentation/GNN_Lecture_60_Slides.pptx) |
| Baseline code (PyTorch + PyTorch Geometric) | [`code/`](code/) |
| Experiment outputs used in the report/slides | [`results/`](results/) |
| Figures used in the report | [`figures/`](figures/) |

## Repository structure

```
.
├── README.md                     this file
├── report/
│   ├── survey.md                 single source of the report (Markdown + LaTeX math)
│   ├── ieee/GNN_Survey_IEEE.tex  IEEE two-column paper (IEEEtran) + ieee/figures/
│   ├── GNN_Literature_Survey.docx   Word version
│   ├── make_ieee.py              survey.md -> IEEEtran LaTeX
│   ├── reference.docx            Word styles used by pandoc
│   └── build_report.sh           rebuilds both the .tex and the .docx
├── presentation/
│   ├── GNN_Lecture_60_Slides.pptx   lecture deck (original .pptx)
│   └── build_deck.js             regenerates the deck from ../results (pptxgenjs)
├── code/
│   ├── models.py                 GCN (baseline) + MLP, SGC, APPNP, GAT controls
│   ├── train.py                  training / validation selection / single test evaluation
│   ├── evaluate.py               reload a checkpoint: per-split metrics, confusion matrix, node predictions
│   ├── run_experiments.py        10-seed study of all models + GCN depth sweep
│   ├── dataset_stats.py          dataset statistics and edge homophily
│   ├── make_figures.py           regenerates every figure in ../figures
│   ├── tests/test_smoke.py       offline unit/smoke tests (synthetic graph)
│   └── requirements.txt
├── results/                      committed copies of the outputs quoted in report and slides
│   ├── summary.csv / summary.json / depth.csv / runs.csv   (run_experiments.py)
│   ├── dataset_stats.json                                  (dataset_stats.py)
│   └── gcn_cora_seed42/{metrics.json,history.csv}          (train.py, seed 42)
├── figures/                      PNG figures embedded in the report
└── archive/v1/                   first version of the report, deck and code (kept for reference)
```

## Baseline algorithm

The two-layer GCN computes

```
Z = softmax( S · ReLU( S · X · W0 ) · W1 ),   S = D̃^(-1/2) (A + I) D̃^(-1/2)
```

and is trained with cross-entropy on the labeled training nodes only. The implementation follows the published Planetoid settings: 16 hidden units, dropout 0.5, Adam, learning rate 0.01, weight decay 5e-4, and 200 epochs. The checkpoint with the best **validation** accuracy is kept, with ties broken by lower validation loss. The **test** mask is evaluated once, after that checkpoint is restored.

To make the survey's methodological claims measurable, the same script also trains four controls under an identical protocol:

| `--model` | Model | Purpose |
|---|---|---|
| `gcn` | Two-layer GCN [Kipf & Welling 2017] | **baseline** |
| `mlp` | Two-layer MLP, no edges | measures the value of the graph |
| `sgc` | SGC, K=2 [Wu et al. 2019] | is the nonlinearity necessary? |
| `appnp` | APPNP, K=10, α=0.1 [Klicpera et al. 2019] | decoupled long-range propagation |
| `gat` | 2-layer GAT, 8 heads [Veličković et al. 2018] | learned neighbor weights |

## Prerequisites

* Python **3.10 or newer** (tested with 3.11)
* pip, and about 1 GB of disk for PyTorch
* Internet access on the first run: PyTorch Geometric downloads the Planetoid files (a few MB) from GitHub
* A GPU is **not** required; everything below runs on a laptop CPU

## Installation

```bash
cd code
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA build of PyTorch, first install the wheel that matches your driver from <https://pytorch.org/get-started/locally/>, then run the `pip install -r requirements.txt` line. Tested versions: torch 2.15.0, torch-geometric 2.8.1, numpy 2.4.6, scipy 1.17.1 (macOS, CPU).

## Usage

All commands are run from `code/` with the virtual environment active.

**1. Offline sanity check (≈5 s, no download)**

```bash
python -m pytest -q tests
```

Expected output: `13 passed`.

**2. Train the GCN baseline**

```bash
python train.py --dataset Cora --seed 42
```

On the first run this downloads Cora into `code/data/`. It logs progress every 20 epochs and writes `results/metrics.json`, `results/history.csv` and `results/best_model.pt`. Reference output on our machine: best epoch 151, validation accuracy 0.798, **test accuracy 0.815**, about 5 s on CPU (see [`results/gcn_cora_seed42/metrics.json`](results/gcn_cora_seed42/metrics.json)). Other hardware or library versions may differ slightly.

Other datasets and models:

```bash
python train.py --dataset CiteSeer --output-dir results_citeseer
python train.py --dataset PubMed   --output-dir results_pubmed
python train.py --model gat   --dataset Cora --output-dir results_gat
python train.py --model sgc   --dataset Cora --output-dir results_sgc
python train.py --num-layers 8 --dataset Cora --output-dir results_gcn8   # deeper plain GCN
python train.py --help                                                    # all options
```

**3. Inference / evaluation demo**

```bash
python evaluate.py --checkpoint results/best_model.pt --nodes 0 1 1708
```

This prints accuracy and macro-F1 on the train, validation and test splits, per-class test accuracy, the test confusion matrix, and the predicted class and confidence for the listed node ids.

**4. Reproduce every table and figure**

```bash
python dataset_stats.py              # Table I    -> results/dataset_stats.json
python run_experiments.py            # Tables IV-V -> results/experiments/  (~35 min on a laptop CPU)
python run_experiments.py --quick    # 2 seeds, Cora only: ~1 min sanity check
python train.py --dataset Cora --seed 42 --output-dir results/gcn_cora_seed42
python make_figures.py --results results/experiments --history results/gcn_cora_seed42/history.csv
```

`make_figures.py` writes to `../figures/`. The committed `results/` folder at the repository root holds the outputs of exactly these commands, and the report and slides quote them.

**5. Rebuild the report and slides (optional)**

```bash
cd report && ./build_report.sh                                       # needs pandoc; writes ieee/*.tex and the .docx
cd presentation && npm install pptxgenjs@3.12.0 && node build_deck.js   # needs Node.js
```

## Compiling the IEEE paper

`report/ieee/` is self-contained: it holds `GNN_Survey_IEEE.tex` and the `figures/` it uses.

* **Overleaf:** zip the `report/ieee` folder, choose *New Project → Upload Project*, and compile with pdfLaTeX. IEEEtran is built in.
* **Local TeX Live / MacTeX:** `cd report/ieee && pdflatex GNN_Survey_IEEE.tex && pdflatex GNN_Survey_IEEE.tex` (run twice so references resolve).

## Results (this project, 10 seeds, public Planetoid split)

| Model | Cora | CiteSeer | PubMed |
|---|---|---|---|
| MLP (no edges) | 58.2 ± 0.8 | 56.7 ± 1.6 | 72.5 ± 0.9 |
| SGC | 80.8 ± 0.6 | 72.2 ± 0.0 | 78.8 ± 0.0 |
| **GCN (baseline)** | 81.9 ± 0.8 | 70.9 ± 1.2 | 79.2 ± 0.3 |
| APPNP | 83.5 ± 0.5 | 71.4 ± 0.5 | 80.0 ± 0.4 |
| GAT | 82.6 ± 1.1 | 72.1 ± 0.7 | 77.5 ± 0.4 |

The full discussion is in Section XV of the report. For comparison, the original GCN paper reports 81.5 / 70.3 / 79.0.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'scipy'` while loading Planetoid | `pip install -r requirements.txt` (scipy is required to read the raw dataset files) |
| Download fails or hangs on first run | Needs network access to `github.com/kimiyoung/planetoid`; or copy an existing `data/<Dataset>/` folder into `code/data/` |
| `RuntimeError` about a corrupt or partial dataset | Delete `code/data/<Dataset>` and rerun |
| `--device cuda requested but CUDA is not available` | Use `--device cpu` or install a CUDA build of PyTorch |
| `No module named torch_geometric` | Make sure the virtual environment is active (`which python`) |
| Slightly different numbers from ours | Expected across hardware and library versions; compare means over seeds |

## Scope and honesty notes

* Every number labelled "this project" comes from the scripts above. Numbers attributed to papers were checked against the papers.
* The experiments are transductive, use a single public split, and use published hyperparameters without tuning. They demonstrate the baseline and the survey's controls; they are not a benchmark ranking.
* GraphSAGE, GIN, GraphSAINT and Geom-GCN are surveyed but not implemented, because their defining settings (inductive, graph classification, very large or heterophilous graphs) differ from Planetoid.

## Credits

Algorithms: Kipf & Welling (GCN), Wu et al. (SGC), Klicpera et al. (APPNP), Veličković et al. (GAT). Framework: PyTorch and PyTorch Geometric (Fey & Lenssen, 2019). Datasets: Planetoid splits (Yang et al., ICML 2016) of Cora/CiteSeer/PubMed (Sen et al., AI Magazine 2008). Full references are in the report.
