"""Phase labels, boundary extraction and scaling fits (Appendix G.2, G.2.1)."""
from pathlib import Path

import numpy as np

GROK, MEMORIZE, FORGET, NOFIT = "grokking", "memorization", "forgetting", "no fitting"
PHASES = (GROK, MEMORIZE, FORGET, NOFIT)


def classify(train_acc, test_acc, thr=0.9):
    """Paper's 90% rules, applied to one run's accuracy curves."""
    tr, te = np.asarray(train_acc, float), np.asarray(test_acc, float)
    tr = np.nan_to_num(tr, nan=0.0)
    te = np.nan_to_num(te, nan=0.0)
    if tr[-1] > thr and te[-1] > thr:
        return GROK
    if tr[-1] > thr:
        return MEMORIZE
    if (tr > thr).any():
        return FORGET
    return NOFIT


def first_crossing(curve, thr, eval_every=100):
    """First update at which `curve` reaches `thr`, or None."""
    idx = np.nonzero(np.asarray(curve, float) >= thr)[0]
    return int(idx[0]) * eval_every if len(idx) else None


def load_sweep(folder):
    """Stack all chunk files into flat arrays."""
    lr, wd, lri, wdi, tr, te = [], [], [], [], [], []
    for f in sorted(Path(folder).glob("wd*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        lr.append(z["lr"]); wd.append(z["wd"]); lri.append(z["lr_idx"]); wdi.append(z["wd_idx"])
        tr.append(z["train_acc"].astype(np.float32)); te.append(z["test_acc"].astype(np.float32))
    if not lr:
        raise FileNotFoundError(f"no sweep chunks in {folder}")
    return dict(lr=np.concatenate(lr), wd=np.concatenate(wd), lr_idx=np.concatenate(lri),
                wd_idx=np.concatenate(wdi), train_acc=np.concatenate(tr), test_acc=np.concatenate(te))


def boundary_points(lr, wd, phase):
    """Per wd column: geometric midpoint of last memorizing lr below the first grokking lr."""
    out = []
    for w in np.unique(wd):
        m = wd == w
        order = np.argsort(lr[m])
        lrs, ph = lr[m][order], np.asarray(phase)[m][order]
        g = np.nonzero(ph == GROK)[0]
        if not len(g):
            continue
        mem = np.nonzero(ph[: g[0]] == MEMORIZE)[0]
        if not len(mem):
            continue
        out.append((w, np.sqrt(lrs[mem[-1]] * lrs[g[0]])))
    return np.array(out).reshape(-1, 2)


def fit_inverse(wd, lr):
    """Fit lr = A / wd (slope fixed at -1 in log-log). Returns A and log-RMSE."""
    logA = np.mean(np.log(lr) + np.log(wd))
    resid = np.log(lr) - (logA - np.log(wd))
    return float(np.exp(logA)), float(np.sqrt(np.mean(resid ** 2)))


def fit_powerlaw(x, y):
    """Least squares log y = c + k log x. Returns slope k, intercept c, R^2."""
    lx, ly = np.log(x), np.log(y)
    k, c = np.polyfit(lx, ly, 1)
    pred = c + k * lx
    r2 = 1 - np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(k), float(c), float(r2)
