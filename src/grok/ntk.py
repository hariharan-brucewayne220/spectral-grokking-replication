"""Closed-form empirical NTKs for the bias-free MLP f(x) = W2^T relu(W1^T x).

With h = relu(W1^T x) and m = 1[W1^T x > 0]:
    d f_c / d W2[k, c'] = delta(c, c') h_k
    d f_c / d W1[i, k]  = W2[k, c] m_k x_i
so
    <grad f_c(x), grad f_d(x')> = delta(c, d) <h, h'> + (x . x') sum_k W2[k,c] W2[k,d] m_k m'_k.
"""
import numpy as np
import torch


def _features(X, W1):
    pre = X @ W1
    return torch.relu(pre), (pre > 0).to(X.dtype)


def full_ntk(X, labels_p, W1, W2):
    """Full kernel K[(u, c), (v, d)] as an [n*p, n*p] matrix (row index = u*p + c)."""
    H, M = _features(X, W1)
    n, p = X.shape[0], W2.shape[1]
    hh = H @ H.T
    xx = X @ X.T
    t2 = torch.einsum("uk,vk,kc,kd->ucvd", M, M, W2, W2) * xx[:, None, :, None]
    t1 = hh[:, None, :, None] * torch.eye(p, dtype=X.dtype)[None, :, None, :]
    return (t1 + t2).reshape(n * p, n * p)


def true_label_ntk(X, y, W1, W2):
    """K_TL[u, v] = <grad f_{y_u}(x_u), grad f_{y_v}(x_v)>  (Appendix G.1, Eq. 105)."""
    H, M = _features(X, W1)
    Wy = W2[:, y].T  # [n, width]: W2[k, y_u]
    t1 = (H @ H.T) * (y[:, None] == y[None, :])
    t2 = ((M * Wy) @ (M * Wy).T) * (X @ X.T)
    return t1 + t2


def label_kernel_raw(K_tl, y, p):
    """Average K_TL over label groups (the matrix K-tilde before Eq. 106). Returns [p, p]."""
    Y = torch.nn.functional.one_hot(y, p).to(K_tl.dtype)
    counts = Y.sum(0)
    Kt = (Y.T @ K_tl @ Y) / (counts[:, None] * counts[None, :])
    return 0.5 * (Kt + Kt.T)


def circulant_spectrum(K):
    """Eigenvalues of the circulant part of K, indexed by Fourier frequency k = 0..p-1."""
    K = np.asarray(K, float)
    p = K.shape[0]
    idx = (np.arange(p)[None, :] - np.arange(p)[:, None]) % p
    row = np.array([K[idx == k].mean() for k in range(p)])
    return np.real(np.fft.fft(row))


def label_kernel(K_tl, y, p):
    """Average K_TL over label groups, then diagonal-normalize (Eq. 106). Returns [p, p]."""
    Kt = label_kernel_raw(K_tl, y, p)
    d = torch.sqrt(torch.diagonal(Kt))
    return Kt / (d[:, None] * d[None, :])


def circulant_fraction(K):
    """Share of the off-diagonal variance of K explained by its best circulant fit.

    A kernel that depends only on (c - c') mod p is circulant; that is the
    Fourier structure modular addition should induce. The diagonal is excluded:
    after diagonal normalization it is all ones, and "ones on the diagonal plus a
    constant elsewhere" is trivially circulant. 1.0 = perfectly circulant,
    0.0 = no off-diagonal variation or none of it circulant.
    """
    K = np.asarray(K, float)
    p = K.shape[0]
    idx = (np.arange(p)[None, :] - np.arange(p)[:, None]) % p
    off = idx != 0
    diag_means = np.array([K[idx == k].mean() for k in range(p)])
    C = diag_means[idx]
    total = np.sum((K[off] - K[off].mean()) ** 2)
    if total < 1e-12 * max(1.0, np.sum(K[off] ** 2)):
        return 0.0
    return float(1 - np.sum((K[off] - C[off]) ** 2) / total)


def circulant_amplitude(K):
    """Std of the circulant (Fourier) part of K over off-diagonal offsets: how strong the structure is."""
    K = np.asarray(K, float)
    p = K.shape[0]
    idx = (np.arange(p)[None, :] - np.arange(p)[:, None]) % p
    return float(np.std([K[idx == k].mean() for k in range(1, p)]))


def fourier_peak(v):
    """Fraction of a real vector's spectral power in its dominant frequency pair (+k, -k)."""
    v = np.asarray(v, float)
    v = v - v.mean()
    P = np.abs(np.fft.fft(v)) ** 2
    p = len(v)
    pair = np.array([P[k] + (P[p - k] if k and p - k != k else 0) for k in range(p // 2 + 1)])
    return float(pair.max() / P.sum()), int(pair.argmax())
