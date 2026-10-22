"""Build docs/explainer.html: the illustrated, kid-friendly explainer, with real data inlined.

Reads results/ and injects one JSON blob into docs/explainer.template.html.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from grok.analysis import FORGET, GROK, MEMORIZE, NOFIT, classify, first_crossing, load_sweep  # noqa: E402
from grok.ntk import circulant_spectrum  # noqa: E402
from grok.sweep import PAPER_LRS, PAPER_WDS  # noqa: E402

M = json.loads((ROOT / "results/metrics.json").read_text())
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()
r2 = lambda xs: [round(float(x), 3) for x in xs]

data = {}

# four reference runs (full resolution: 241 points every 100 updates)
data["ref"] = {}
for name in ("memorization", "grokking", "forgetting", "no fitting"):
    z = np.load(ROOT / "results/reference" / f"{name.replace(' ', '_')}.npz")
    data["ref"][name] = dict(tr=r2(z["train_acc"]), te=r2(z["test_acc"]))
data["ref_summary"] = M["reference_cells"]

# the sweep: phase grid + every run's curves (every 4th eval point, uint8)
S = load_sweep(ROOT / "results/mlp23")
li_all, wi_all = np.unique(S["lr_idx"]), np.unique(S["wd_idx"])
L, W = len(li_all), len(wi_all)
code = {GROK: 0, MEMORIZE: 1, FORGET: 2, NOFIT: 3}
phase = np.full((W, L), -1, int)
s10 = np.full((W, L), -1, int)
curves = np.zeros((W, L, 2, 61), np.uint8)
for k, (li, wi) in enumerate(zip(S["lr_idx"], S["wd_idx"])):
    r, c = int(np.searchsorted(wi_all, wi)), int(np.searchsorted(li_all, li))
    tr = np.nan_to_num(S["train_acc"][k].astype(float))
    te = np.nan_to_num(S["test_acc"][k].astype(float))
    ph = classify(tr, te)
    phase[r, c] = code[ph]
    t = first_crossing(te, 0.10)
    s10[r, c] = t if (t is not None and ph == GROK) else -1
    curves[r, c, 0] = np.round(tr[::4] * 255)
    curves[r, c, 1] = np.round(te[::4] * 255)
data["grid"] = dict(lrs=[float(x) for x in PAPER_LRS[li_all]], wds=[float(x) for x in PAPER_WDS[wi_all]],
                    phase=phase.tolist(), s10=s10.tolist(), curves=b64(curves), n_runs=int(len(S["lr"])))
data["sweep"] = M.get("sweep", {})
data["boundary"] = M.get("boundary", {})
data["timing"] = M.get("timing_law", {})

# NTK "friends map": diagonal-hidden normalized label kernels at each checkpoint
z = np.load(ROOT / "results/ntk97/label_kernels.npz")
steps = z["steps"].tolist()
frames, task = [], []
for K, Kr in zip(z["kernels"], z["raw_kernels"]):
    K = K.copy()
    np.fill_diagonal(K, np.nan)
    K = K - np.nanmean(K)
    K = K / np.nanmax(np.abs(K))
    K = np.nan_to_num(K, nan=1.0)
    frames.append(np.round((K + 1) * 127.5))
    sp = circulant_spectrum(Kr)[1:]
    task.append(float(sp.max() - np.median(sp)))
acc = np.loadtxt(ROOT / "results/ntk97/accuracy.csv", delimiter=",", skiprows=1)
data["ntk"] = dict(steps=steps, frames=b64(np.stack(frames)), p=int(z["kernels"].shape[1]), task=r2([t * 1000 for t in task]),
                   acc_step=acc[:, 0].astype(int).tolist(), acc_tr=r2(acc[:, 1]), acc_te=r2(acc[:, 2]),
                   fit_step=M["ntk97"]["first_train_100"])

# seeds
seeds = json.loads((ROOT / "results/seeds/summary.json").read_text())
data["seeds"] = seeds

# ridge (summary only)
data["ridge"] = {k: {s: v[s]["relative_error_vs_ridge_prediction"] for s in v if s.startswith("after_")}
                 for k, v in M["ridge"].items()}

tmpl = (ROOT / "docs/explainer.template.html").read_text()
out = tmpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
(ROOT / "docs/explainer.html").write_text(out)
print(f"explainer: {len(out) / 1e6:.2f} MB, {data['grid']['n_runs']} runs, grid {W}x{L}")
