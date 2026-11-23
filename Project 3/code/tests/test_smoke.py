"""Fast offline checks (no dataset download): python -m pytest -q tests"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import DEFAULTS, build_model  # noqa: E402
from train import macro_f1, masked_accuracy, set_seed  # noqa: E402


def toy_graph(n=40, d=12, c=3, seed=0):
    """Two-community-style random graph whose labels are predictable from features."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % c
    x = torch.randn(n, d, generator=g) + F.one_hot(y, d).float() * 3
    src = torch.randint(0, n, (4 * n,), generator=g)
    dst = torch.randint(0, n, (4 * n,), generator=g)
    edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    return x, edge_index, y


@pytest.mark.parametrize('name', sorted(DEFAULTS))
def test_forward_shape(name):
    x, ei, y = toy_graph()
    h = DEFAULTS[name]['hidden'] or 8
    model = build_model(name, x.size(1), 3, hidden=h, dropout=0.5)
    model.eval()
    out = model(x, ei)
    assert out.shape == (x.size(0), 3)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('name', sorted(DEFAULTS))
def test_training_reduces_loss(name):
    set_seed(0)
    x, ei, y = toy_graph()
    cfg = DEFAULTS[name]
    model = build_model(name, x.size(1), 3, hidden=cfg['hidden'] or 8, dropout=cfg['dropout'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    losses = []
    for _ in range(30):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(x, ei), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_seed_reproducible():
    x, ei, _ = toy_graph()
    outs = []
    for _ in range(2):
        set_seed(123)
        model = build_model('gcn', x.size(1), 3, hidden=16, dropout=0.5)
        model.eval()
        outs.append(model(x, ei))
    assert torch.equal(outs[0], outs[1])


def test_deep_gcn_builds():
    x, ei, _ = toy_graph()
    model = build_model('gcn', x.size(1), 3, hidden=16, dropout=0.0, num_layers=8)
    assert len(model.convs) == 8
    assert model(x, ei).shape == (x.size(0), 3)


def test_metrics():
    labels = torch.tensor([0, 0, 1, 1])
    logits = torch.tensor([[2., 0], [0, 2], [0, 2], [0, 2]])
    mask = torch.tensor([True, True, True, False])
    assert masked_accuracy(logits, labels, mask) == pytest.approx(2 / 3)
    pred = logits.argmax(-1)
    # class0: tp1 fp0 fn1 -> 2/3 ; class1: tp2 fp1 fn0 -> 4/5
    assert macro_f1(pred, labels, 2) == pytest.approx((2 / 3 + 4 / 5) / 2)
    with pytest.raises(ValueError):
        masked_accuracy(logits, labels, torch.zeros(4, dtype=torch.bool))
