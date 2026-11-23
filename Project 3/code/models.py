"""Model definitions for the GNN node-classification baseline and its controls.

The primary baseline is the two-layer GCN of Kipf and Welling [1]. The other
models are the controlled comparisons discussed in the survey:

* ``mlp``   -- feature-only control (no edges are used at all)
* ``sgc``   -- Simple Graph Convolution, Wu et al. [5]: softmax(S^K X Theta)
* ``appnp`` -- predict-then-propagate, Klicpera et al. [6]
* ``gat``   -- graph attention network, Velickovic et al. [3]

Every model exposes the same ``forward(x, edge_index) -> logits`` interface so
that ``train.py`` can train and evaluate all of them with identical code.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import APPNP as APPNPPropagation
from torch_geometric.nn import GATConv, GCNConv, SGConv


class GCN(nn.Module):
    """Kipf--Welling GCN: H^(l+1) = sigma(S H^(l) W^(l)), S = D^-1/2 (A+I) D^-1/2.

    ``num_layers=2`` reproduces the architecture of [1]. Larger values are used
    only by the depth (over-smoothing) experiment; no residual connections or
    normalisation layers are added, so deep variants behave as plain stacks.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float,
                 num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError('num_layers must be >= 1')
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class MLP(nn.Module):
    """Two-layer perceptron on node features only: the 'is the graph useful?' control."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index=None):  # edge_index intentionally ignored
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)


class SGC(nn.Module):
    """Simple Graph Convolution [5]: a linear classifier on S^K X.

    ``cached=True`` makes PyG compute S^K X once (first forward pass) and reuse
    it, which is exactly the precomputation argument made in [5]. Because the
    cache is tied to one graph, a fresh model must be built per dataset.
    """

    def __init__(self, in_dim: int, out_dim: int, K: int = 2):
        super().__init__()
        self.conv = SGConv(in_dim, out_dim, K=K, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class APPNPNet(nn.Module):
    """Predict-then-propagate [6]: H0 = MLP(X); H(k+1) = (1-a) S H(k) + a H0."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float,
                 K: int = 10, alpha: float = 0.1):
        super().__init__()
        self.mlp = MLP(in_dim, hidden, out_dim, dropout)
        self.prop = APPNPPropagation(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        return self.prop(self.mlp(x), edge_index)


class GAT(nn.Module):
    """Two-layer graph attention network following the transductive setup of [3]."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float,
                 heads: int = 8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


# Default hyperparameters. GCN/GAT/SGC values follow the original papers'
# Planetoid settings; APPNP uses the K and alpha reported in [6] but a single
# weight decay for all layers. None of these were tuned on the test set.
DEFAULTS = {
    'gcn':   dict(hidden=16, dropout=0.5, lr=0.01,  weight_decay=5e-4, epochs=200, patience=100),
    'mlp':   dict(hidden=16, dropout=0.5, lr=0.01,  weight_decay=5e-4, epochs=200, patience=100),
    'sgc':   dict(hidden=0,  dropout=0.0, lr=0.2,   weight_decay=5e-5, epochs=100, patience=100),
    'appnp': dict(hidden=64, dropout=0.5, lr=0.01,  weight_decay=5e-4, epochs=500, patience=100),
    'gat':   dict(hidden=8,  dropout=0.6, lr=0.005, weight_decay=5e-4, epochs=500, patience=100),
}


def build_model(name: str, in_dim: int, out_dim: int, hidden: int, dropout: float,
                num_layers: int = 2, K: int | None = None, alpha: float = 0.1,
                heads: int = 8) -> nn.Module:
    """Factory used by train.py and evaluate.py."""
    if name == 'gcn':
        return GCN(in_dim, hidden, out_dim, dropout, num_layers)
    if name == 'mlp':
        return MLP(in_dim, hidden, out_dim, dropout)
    if name == 'sgc':
        return SGC(in_dim, out_dim, K=K or 2)
    if name == 'appnp':
        return APPNPNet(in_dim, hidden, out_dim, dropout, K=K or 10, alpha=alpha)
    if name == 'gat':
        return GAT(in_dim, hidden, out_dim, dropout, heads=heads)
    raise ValueError(f'Unknown model {name!r}; choose from {sorted(DEFAULTS)}')
