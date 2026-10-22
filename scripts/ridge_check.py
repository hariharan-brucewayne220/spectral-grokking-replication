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

summary = {}
for name, (lr, wd) in CELLS.items():
    _, _, model = train_reference(d, lr, wd, 24_000)
    W1 = model[0].weight.detach().T.double()
    W2 = model[2].weight.detach().T.double()
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
    # stationarity check: |grad of objective| relative to |data-gradient term|
    params = [model[0].weight, model[2].weight]
    loss = torch.nn.functional.mse_loss(model(Xtr), torch.nn.functional.one_hot(ytr, p).float())
    g = torch.autograd.grad(loss, params)
    full_g = torch.cat([(gi + wd * pi).reshape(-1) for gi, pi in zip(g, params)]).norm().item()
    data_g = torch.cat([gi.reshape(-1) for gi in g]).norm().item()
    keep = yj.abs() > 1e-3 * yj.abs().max()
    np.savez_compressed(out / f"{name}.npz", evals=evals.numpy(), yj=yj.numpy(), sj=sj.numpy(), c=c)
    summary[name] = dict(
        lr=lr, wd=wd, c=c, residual_norm=sigma.norm().item(), target_norm=Y.norm().item(),
        relative_error_vs_ridge_prediction=rel_err,
        grad_norm_ratio=full_g / data_g,
        corr_measured_vs_predicted_ratio=float(np.corrcoef(
            (sj[keep] / yj[keep]).numpy(), pred_ratio[keep].numpy())[0, 1]),
        n_directions_compared=int(keep.sum()),
    )
    print(name, summary[name], flush=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
