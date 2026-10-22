"""Claim C5 (Fig. 2): Fourier structure appears in the label-space NTK after training fits.

Appendix G.1: mod 97, 4,753 unordered pairs, 50/50 split (NumPy seed 0), bias-free
194 -> 256 -> 97 ReLU MLP, He-normal seed 0, full-batch SGD lr 40, wd 2.5e-5, 20,000 updates.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok.data import unordered_random_split  # noqa: E402
from grok.model import make_mlp  # noqa: E402
from grok.ntk import circulant_fraction, fourier_peak, label_kernel, true_label_ntk  # noqa: E402

torch.set_num_threads(1)
P, LR, WD, STEPS = 97, 40.0, 2.5e-5, 20_000
CHECKPOINTS = [0, 100, 500, 1000, 3000, 6000, 8000, 10000, 20000]

d = unordered_random_split(P, seed=0)
Xtr, ytr, Xte, yte = d.split()
Ytr = nn.functional.one_hot(ytr, P).float()
model = make_mlp(2 * P, 256, P, seed=0)
opt = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WD, momentum=0)

out = Path("results/ntk97")
out.mkdir(parents=True, exist_ok=True)
kernels, log = {}, []
t0 = time.time()
for s in range(STEPS + 1):
    if s % 100 == 0:
        with torch.no_grad():
            tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
            te = (model(Xte).argmax(1) == yte).float().mean().item()
        log.append((s, tr, te))
    if s in CHECKPOINTS:
        W1 = model[0].weight.detach().T.double()
        W2 = model[2].weight.detach().T.double()
        Kbar = label_kernel(true_label_ntk(Xtr.double(), ytr, W1, W2), ytr, P).numpy()
        kernels[s] = Kbar
        evals, evecs = np.linalg.eigh(Kbar)
        top = [fourier_peak(evecs[:, -1 - i]) for i in range(4)]
        print(f"step {s:6d} train {tr:.3f} test {te:.3f} circulant {circulant_fraction(Kbar):.3f} "
              f"top-4 eigvec Fourier peak {[round(f, 2) for f, _ in top]} ({time.time() - t0:.0f}s)", flush=True)
    if s == STEPS:
        break
    opt.zero_grad()
    nn.functional.mse_loss(model(Xtr), Ytr).backward()
    opt.step()

np.savez_compressed(out / "label_kernels.npz", steps=np.array(list(kernels)),
                    kernels=np.stack(list(kernels.values())))
np.savetxt(out / "accuracy.csv", np.array(log), delimiter=",", header="step,train_acc,test_acc", comments="")
summary = {}
for s, K in kernels.items():
    evals, evecs = np.linalg.eigh(K)
    summary[s] = dict(
        circulant_fraction=circulant_fraction(K),
        top_eigvec_fourier_peak=[fourier_peak(evecs[:, -1 - i])[0] for i in range(4)],
        top_eigvec_frequency=[fourier_peak(evecs[:, -1 - i])[1] for i in range(4)],
    )
(out / "summary.json").write_text(json.dumps(summary, indent=2))
