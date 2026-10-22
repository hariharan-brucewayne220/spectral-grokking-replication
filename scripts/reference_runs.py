"""Claims C0/C1: the four representative mod-23 cells of Fig. 7 (Appendix G.2).

Uses the plain nn.Module + torch.optim.SGD trainer, independent of the batched sweep code.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok.analysis import classify, first_crossing  # noqa: E402
from grok.data import difference_split  # noqa: E402
from grok.train import train_reference  # noqa: E402

torch.set_num_threads(1)
CELLS = {
    "memorization": dict(lr=0.87048, wd=3.2930e-5, paper="memorization"),
    "grokking": dict(lr=2.36316, wd=1.18329e-4, paper="grokking; first 90% held-out at update 8,300"),
    "forgetting": dict(lr=1.43426, wd=1.24846e-3, paper="forgetting; peak train acc 0.948, final 0.204"),
    "no fitting": dict(lr=28.6967, wd=1.89561e-4, paper="no fitting"),
}

out = Path("results/reference")
out.mkdir(parents=True, exist_ok=True)
summary = {}
for name, c in CELLS.items():
    tr, te, _ = train_reference(difference_split(), c["lr"], c["wd"], 24_000)
    tr, te = tr.numpy(), te.numpy()
    np.savez_compressed(out / f"{name.replace(' ', '_')}.npz", train_acc=tr, test_acc=te)
    summary[name] = dict(
        lr=c["lr"], wd=c["wd"], paper=c["paper"], phase=classify(tr, te),
        first_test_90=first_crossing(te, 0.9), peak_train=float(np.nanmax(tr)),
        final_train=float(tr[-1]), final_test=float(te[-1]),
    )
    print(name, summary[name], flush=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
