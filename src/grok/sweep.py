"""Resumable (lr, wd) phase sweep for the mod-23 MLP (Appendix G.2).

The paper's grid is 84 log-spaced learning rates in [0.1, 100] and 90 log-spaced
weight decays in [5e-6, 2e-3]. `--stride 2` keeps every second point of that
exact grid (42 x 45 = 1,890 runs), so our cells coincide with cells of theirs.

Each chunk of runs is saved to its own .npz; finished chunks are skipped on restart.

    python -m grok.sweep --out results/mlp23 --stride 2 --workers 3
"""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PAPER_LRS = np.logspace(np.log10(0.1), np.log10(100), 84)
PAPER_WDS = np.logspace(np.log10(5e-6), np.log10(2e-3), 90)
STEPS = 24_000
EVAL_EVERY = 100


def _run_chunk(job):
    out, name, lr_idx, wd_idx = job
    import torch
    torch.set_num_threads(1)
    from grok.data import difference_split
    from grok.train import train_batched

    lrs = PAPER_LRS[lr_idx]
    wds = PAPER_WDS[wd_idx]
    t0 = time.time()
    tr, te = train_batched(difference_split(), lrs, wds, STEPS, eval_every=EVAL_EVERY)
    path = Path(out) / f"{name}.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, lr_idx=lr_idx, wd_idx=wd_idx, lr=lrs, wd=wds,
                        train_acc=tr.numpy().astype(np.float16),
                        test_acc=te.numpy().astype(np.float16))
    os.replace(tmp, path)
    return name, time.time() - t0


def build_jobs(out, stride, chunk, offset=0):
    jobs = []
    lr_all = np.arange(offset, 84, stride)
    for wi in range(offset, 90, stride):
        for c in range(0, len(lr_all), chunk):
            li = lr_all[c:c + chunk]
            name = f"wd{wi:02d}_lr{li[0]:02d}-{li[-1]:02d}"
            if not (Path(out) / f"{name}.npz").exists():
                jobs.append((out, name, li, np.full(len(li), wi)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mlp23")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=7)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.out, args.stride, args.chunk, args.offset)
    print(f"{len(jobs)} chunks to run", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, dt) in enumerate(pool.imap_unordered(_run_chunk, jobs), 1):
            el = time.time() - t0
            eta = el / i * (len(jobs) - i)
            print(f"[{i}/{len(jobs)}] {name} {dt:.0f}s | elapsed {el/60:.1f}m eta {eta/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
