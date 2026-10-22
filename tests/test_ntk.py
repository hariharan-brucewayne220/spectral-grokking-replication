import numpy as np
import torch
from torch.func import functional_call, jacrev

from grok.data import difference_split
from grok.model import make_mlp
from grok.ntk import circulant_fraction, fourier_peak, full_ntk, label_kernel, true_label_ntk


def _autograd_jacobian(m, X):
    params = {k: v.detach() for k, v in m.named_parameters()}
    jac = jacrev(lambda ps: functional_call(m, ps, (X,)))(params)
    n, p = X.shape[0], m[2].out_features
    return torch.cat([j.reshape(n, p, -1) for j in jac.values()], dim=-1)  # [n, p, P]


def _setup(n=12):
    torch.manual_seed(0)
    m = make_mlp(46, 64, 23, seed=5).double()
    d = difference_split()
    X, y = d.X.double()[:n], d.y[:n]
    W1, W2 = m[0].weight.detach().T, m[2].weight.detach().T
    return m, X, y, W1, W2


def test_full_ntk_matches_autograd():
    m, X, y, W1, W2 = _setup()
    J = _autograd_jacobian(m, X).reshape(X.shape[0] * 23, -1)
    torch.testing.assert_close(full_ntk(X, None, W1, W2), J @ J.T)


def test_true_label_ntk_matches_autograd():
    m, X, y, W1, W2 = _setup()
    J = _autograd_jacobian(m, X)
    Jy = J[torch.arange(len(y)), y]
    torch.testing.assert_close(true_label_ntk(X, y, W1, W2), Jy @ Jy.T)


def test_label_kernel_unit_diagonal():
    m, X, y, W1, W2 = _setup(200)
    K = label_kernel(true_label_ntk(X, y, W1, W2), y, 23)
    assert torch.allclose(torch.diagonal(K), torch.ones(23, dtype=K.dtype))


def test_circulant_fraction_and_fourier_peak():
    p = 23
    idx = (np.arange(p)[None, :] - np.arange(p)[:, None]) % p
    C = np.cos(2 * np.pi * 3 * idx / p)
    assert circulant_fraction(C) > 0.999
    rng = np.random.default_rng(0)
    R = rng.normal(size=(p, p)); R = R + R.T
    assert circulant_fraction(R) < 0.3
    # identity plus a constant is trivially circulant; it must not count as structure
    assert circulant_fraction(0.3 * np.ones((p, p)) + 0.7 * np.eye(p)) == 0.0
    noisy = 0.3 + 0.7 * np.eye(p) + 0.01 * R
    assert circulant_fraction(noisy) < 0.3
    frac, k = fourier_peak(np.cos(2 * np.pi * 5 * np.arange(p) / p))
    assert k == 5 and frac > 0.99
