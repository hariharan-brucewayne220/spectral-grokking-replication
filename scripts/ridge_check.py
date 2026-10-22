"""Claim C6 (Theorem 1): the post-fit residual is a ridge residual.

For a 2-homogeneous net at a stationary point of  mean((f-y)^2) + (wd/2)|theta|^2,
multiplying the stationarity condition by J and using J theta = 2 f gives
    sigma* = -c (K + c I)^{-1} y,   c = wd * n * p   (= D * lambda_norm in the paper),
so in each NTK eigendirection j:  sigma_j / y_j = -c / (Lambda_j + c).
Weakly represented directions (small Lambda_j) keep a larger share of their target as error.

We train the Fig. 7 grokking and memorization cells for 24,000 updates, then compare the
measured residual with this prediction using the network's own empirical NTK.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok.data import difference_split  # noqa: E402
from grok.ntk import full_ntk  # noqa: E402
from grok.train import train_reference  # noqa: E402

torch.set_num_threads(1)
CELLS = {"grokking": (2.36316, 1.18329e-4), "memorization": (0.87048, 3.2930e-5)}
out = Path("results/ridge")
out.mkdir(parents=True, exist_ok=True)
d = difference_split()
Xtr, ytr, _, _ = d.split()
n, p = len(ytr), d.p
Y = torch.nn.functional.one_hot(ytr, p).double().reshape(-1)

def analyze(W1, W2, wd, tag):
    """Compare the residual at weights (W1, W2) with sigma* computed from their own NTK."""
    X = Xtr.double()
    f = (torch.relu(X @ W1) @ W2).reshape(-1)
    sigma = f - Y
    K = full_ntk(X, None, W1, W2)
    c = wd * n * p
    evals, U = torch.linalg.eigh(K)
    yj, sj = U.T @ Y, U.T @ sigma
    pred_ratio = -c / (evals + c)
    sigma_star = U @ (pred_ratio * yj)
    rel_err = (torch.linalg.norm(sigma - sigma_star) / torch.linalg.norm(sigma)).item()
    W1g, W2g = W1.clone().requires_grad_(), W2.clone().requires_grad_()
    loss = ((torch.relu(X @ W1g) @ W2g - Y.view(n, p)) ** 2).mean()
    g1, g2 = torch.autograd.grad(loss, (W1g, W2g))
    full_g = torch.cat([(g1 + wd * W1).reshape(-1), (g2 + wd * W2).reshape(-1)]).norm().item()
    data_g = torch.cat([g1.reshape(-1), g2.reshape(-1)]).norm().item()
    keep = yj.abs() > 1e-3 * yj.abs().max()
    np.savez_compressed(out / f"{tag}.npz", evals=evals.numpy(), yj=yj.numpy(), sj=sj.numpy(), c=c)
    return dict(c=c, residual_norm=sigma.norm().item(), target_norm=Y.norm().item(),
                relative_error_vs_ridge_prediction=rel_err, grad_norm_ratio=full_g / data_g,
                corr_measured_vs_predicted_ratio=float(np.corrcoef(
                    (sj[keep] / yj[keep]).numpy(), pred_ratio[keep].numpy())[0, 1]),
                n_directions_compared=int(keep.sum()))


def settle(W1, W2, wd, iters=3000):
    """Polish to a nearby stationary point of mean((f-y)^2) + wd/2 |theta|^2 with L-BFGS in float64."""
    W1 = W1.contiguous().clone().requires_grad_()
    W2 = W2.contiguous().clone().requires_grad_()
    X, Yt = Xtr.double(), Y.view(n, p)
    opt = torch.optim.LBFGS([W1, W2], lr=1, max_iter=iters, tolerance_grad=1e-14,
                            tolerance_change=1e-16, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        obj = ((torch.relu(X @ W1) @ W2 - Yt) ** 2).mean() + 0.5 * wd * (W1.pow(2).sum() + W2.pow(2).sum())
        obj.backward()
        return obj
    opt.step(closure)
    return W1.detach(), W2.detach()


summary = {}
for name, (lr, wd) in CELLS.items():
    _, _, model = train_reference(d, lr, wd, 24_000)
    W1 = model[0].weight.detach().T.double()
    W2 = model[2].weight.detach().T.double()
    summary[name] = dict(lr=lr, wd=wd, after_24k_sgd=analyze(W1, W2, wd, name))
    print(name, "after SGD", summary[name]["after_24k_sgd"], flush=True)
    W1s, W2s = settle(W1, W2, wd)
    summary[name]["after_settling"] = analyze(W1s, W2s, wd, f"{name}_settled")
    print(name, "settled", summary[name]["after_settling"], flush=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
