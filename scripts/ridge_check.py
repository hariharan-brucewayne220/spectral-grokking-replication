"""Claim C6 (Theorem 1): the post-fit residual is a ridge residual.

For a 2-homogeneous net at a stationary point of  mean((f-y)^2) + (wd/2)|theta|^2,
multiplying the stationarity condition by J and using J theta = 2 f gives
    sigma* = -c (K + c I)^{-1} y,   c = wd * n * p   (= D * lambda_norm in the paper),
so in each NTK eigendirection j:  sigma_j / y_j = -c / (Lambda_j + c).
Weakly represented directions (small Lambda_j) keep a larger share of their target as error.

We train the Fig. 7 grokking and memorization cells and compare the measured residual with
this prediction, using the network's own empirical NTK, after 24k, 100k and 300k updates.
(An earlier version tried to reach stationarity with L-BFGS; it stalled at ReLU kinks with
|grad objective| / |grad data| still ~0.37, so we dropped it.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok.data import difference_split  # noqa: E402
from grok.ntk import full_ntk  # noqa: E402
from grok.model import make_mlp  # noqa: E402

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


CHECKS = (24_000, 100_000, 300_000)


summary = {}
for name, (lr, wd) in CELLS.items():
    # Keep training with the same SGD; the theory says the net approaches stationarity
    # on the slow clock 1/(lr*wd), so the ridge relation should tighten over time.
    model = make_mlp(2 * p, 256, p, seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0)
    Yt = torch.nn.functional.one_hot(ytr, p).float()
    summary[name] = dict(lr=lr, wd=wd, slow_clock_updates=1 / (lr * wd))
    done = 0
    for target in CHECKS:
        for _ in range(target - done):
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(Xtr), Yt).backward()
            opt.step()
        done = target
        W1 = model[0].weight.detach().T.double()
        W2 = model[2].weight.detach().T.double()
        summary[name][f"after_{target}"] = analyze(W1, W2, wd, f"{name}_{target}")
        print(name, target, summary[name][f"after_{target}"], flush=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
