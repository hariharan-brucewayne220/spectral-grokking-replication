"""Robustness: does the 1/(lr*wd) timing hold across model seeds, not just seed 0?

The paper's sweep uses model seed 0 everywhere. We take three grokking cells that lie
along a line of decreasing lr*wd and train 5 seeds each.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok.analysis import classify, first_crossing  # noqa: E402
from grok.data import difference_split  # noqa: E402
from grok.train import train_batched  # noqa: E402

torch.set_num_threads(1)
CELLS = [(2.36316, 1.18329e-4), (4.0, 1.18329e-4), (2.36316, 2.4e-4)]
SEEDS = range(5)
d = difference_split()
out = Path("results/seeds")
out.mkdir(parents=True, exist_ok=True)
rows = []
for seed in SEEDS:
    tr, te = train_batched(d, [c[0] for c in CELLS], [c[1] for c in CELLS], 24_000, seed=seed)
    for (lr, wd), a, b in zip(CELLS, tr.numpy(), te.numpy()):
        s10, s90 = first_crossing(b, 0.10), first_crossing(b, 0.90)
        rows.append(dict(seed=seed, lr=lr, wd=wd, phase=classify(a, b), s10=s10, s90=s90,
                         lr_wd_s10=None if s10 is None else lr * wd * s10))
        print(rows[-1], flush=True)
(out / "summary.json").write_text(json.dumps(rows, indent=2))
