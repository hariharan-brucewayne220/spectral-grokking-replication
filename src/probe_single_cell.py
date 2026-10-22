"""Feasibility probe: one cell of the paper's mod-23 MLP sweep (Appendix G.2).

Paper says the grokking example (lr=2.36316, wd=1.18329e-4) first hits 90%
held-out accuracy at update 8,300. We check timing and whether we land close.
"""
import sys, time
import numpy as np
import torch
import torch.nn as nn

P = 23
D = {0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 16, 17, 19, 20, 22}
lr, wd, steps = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(1)

pairs = [(a, b) for a in range(P) for b in range(P)]
X = torch.zeros(len(pairs), 2 * P)
y = torch.tensor([(a + b) % P for a, b in pairs])
for i, (a, b) in enumerate(pairs):
    X[i, a] = 1
    X[i, P + b] = 1
train = torch.tensor([((a - b) % P) in D for a, b in pairs])
Xtr, ytr, Xte, yte = X[train], y[train], X[~train], y[~train]
Ytr = nn.functional.one_hot(ytr, P).float()
print("train/test", len(ytr), len(yte))

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2 * P, 256, bias=False), nn.ReLU(), nn.Linear(256, P, bias=False))
for m in model:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0)

t0 = time.time()
first90 = None
for s in range(steps + 1):
    if s % 100 == 0:
        with torch.no_grad():
            tr = (model(Xtr).argmax(1) == ytr).float().mean().item()
            te = (model(Xte).argmax(1) == yte).float().mean().item()
        if first90 is None and te >= 0.9:
            first90 = s
        if s % 2000 == 0:
            print(f"step {s:6d} train {tr:.3f} test {te:.3f}")
    if s == steps:
        break
    opt.zero_grad()
    nn.functional.mse_loss(model(Xtr), Ytr).backward()
    opt.step()
print(f"first 90% held-out at {first90}; wall {time.time() - t0:.1f}s")
