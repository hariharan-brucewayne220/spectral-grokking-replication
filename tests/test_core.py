import numpy as np
import pytest
import torch
from torch.func import functional_call, jvp

from grok.analysis import FORGET, GROK, MEMORIZE, NOFIT, boundary_points, classify, first_crossing, fit_inverse
from grok.data import difference_split, unordered_random_split
from grok.model import make_mlp
from grok.train import train_batched, train_reference


def test_difference_split_sizes():
    d = difference_split()
    assert d.train.sum().item() == 368
    assert (~d.train).sum().item() == 161


def test_difference_split_is_shift_invariant():
    d = difference_split()
    p = d.p
    in_train = {(a, b) for a, b, t in zip(d.a.tolist(), d.b.tolist(), d.train.tolist()) if t}
    for r in range(p):
        shifted = {((a + r) % p, (b + r) % p) for a, b in in_train}
        assert shifted == in_train


def test_unordered_split_sizes():
    d = unordered_random_split(97, seed=0)
    assert len(d.y) == 4753
    assert d.train.sum().item() == 2376


def test_labels_are_mod_sum():
    d = difference_split()
    assert torch.equal(d.y, (d.a + d.b) % 23)
    X = d.X
    assert torch.equal(X.sum(1), torch.full((529,), 2.0))


def test_model_is_bias_free_with_expected_size():
    m = make_mlp(46, 256, 23)
    assert all(mod.bias is None for mod in m if isinstance(mod, torch.nn.Linear))
    assert sum(p.numel() for p in m.parameters()) == 46 * 256 + 256 * 23 == 17_664


def test_two_homogeneity_and_euler_identity():
    """f(c theta) = c^2 f(theta) and J theta = 2 f: the property the theory is built on."""
    torch.manual_seed(1)
    m = make_mlp(46, 256, 23, seed=3).double()
    X = difference_split().X.double()
    params = {k: v.detach() for k, v in m.named_parameters()}
    f = lambda ps: functional_call(m, ps, (X,))
    base = f(params)
    for c in (0.5, 1.7, 3.0):
        scaled = f({k: c * v for k, v in params.items()})
        torch.testing.assert_close(scaled, c ** 2 * base)
    _, j_theta = jvp(f, (params,), (params,))
    torch.testing.assert_close(j_theta, 2 * base)


@pytest.mark.parametrize("lr,wd", [(2.36316, 1.18329e-4), (0.5, 1e-3), (30.0, 2e-5)])
def test_batched_matches_reference(lr, wd):
    d = difference_split()
    steps = 400
    tr_ref, te_ref, model = train_reference(d, lr, wd, steps)
    tr_b, te_b = train_batched(d, [lr, lr], [wd, wd], steps)
    for k in range(2):
        np.testing.assert_allclose(tr_b[k].numpy(), tr_ref.numpy(), atol=1 / 368 + 1e-6)
        np.testing.assert_allclose(te_b[k].numpy(), te_ref.numpy(), atol=1 / 161 + 1e-6)


def test_batched_runs_are_independent():
    """A model's trajectory must not depend on which other (lr, wd) share its batch."""
    d = difference_split()
    alone = train_batched(d, [2.0], [1e-4], 300)[1][0]
    mixed = train_batched(d, [50.0, 2.0, 0.1], [1e-3, 1e-4, 5e-6], 300)[1][1]
    torch.testing.assert_close(alone, mixed)


def test_deterministic():
    d = difference_split()
    a = train_batched(d, [2.0], [1e-4], 300)
    b = train_batched(d, [2.0], [1e-4], 300)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_classify_rules():
    up = np.linspace(0, 1, 50)
    assert classify(up, up) == GROK
    assert classify(up, np.full(50, 0.3)) == MEMORIZE
    assert classify(np.r_[up, np.full(10, 0.2)], np.zeros(60)) == FORGET
    assert classify(np.full(50, 0.5), np.zeros(50)) == NOFIT
    assert classify(np.full(50, np.nan), np.full(50, np.nan)) == NOFIT


def test_first_crossing():
    c = np.array([0.0, 0.05, 0.2, 0.95, 1.0])
    assert first_crossing(c, 0.1) == 200
    assert first_crossing(c, 0.9) == 300
    assert first_crossing(c, 1.1) is None


def test_boundary_and_inverse_fit_recover_known_law():
    lrs = np.logspace(-1, 2, 40)
    wds = np.logspace(-5, -3, 15)
    A = 1e-4
    LR, WD = np.meshgrid(lrs, wds)
    phase = np.where(LR > A / WD, GROK, MEMORIZE)
    pts = boundary_points(LR.ravel(), WD.ravel(), phase.ravel())
    fitA, rmse = fit_inverse(pts[:, 0], pts[:, 1])
    assert abs(np.log(fitA / A)) < 0.1
