"""Build every figure and results/metrics.json from saved results.

Works on a partial sweep too (missing cells are left blank).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grok.analysis import (FORGET, GROK, MEMORIZE, NOFIT, boundary_points, classify,  # noqa: E402
                           first_crossing, fit_inverse, fit_powerlaw, load_sweep)
from grok.ntk import circulant_amplitude, circulant_fraction, circulant_spectrum, fourier_peak  # noqa: E402
from grok.sweep import PAPER_LRS, PAPER_WDS  # noqa: E402

FIG, RES = Path("figures"), Path("results")
FIG.mkdir(exist_ok=True)

# Reference palette (dataviz skill, light mode). No-fit is a neutral, not a hue.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
C_GROK, C_MEM, C_FORGET, C_NOFIT = "#2a78d6", "#eb6834", "#1baf7a", "#d9d7d0"
PAPER_A, PAPER_SLOPE = 1.0916395e-4, -1.1681

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
})
metrics = {}


def reference_figure():
    path = RES / "reference" / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    metrics["reference_cells"] = summary
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, (name, s) in zip(axes, summary.items()):
        z = np.load(RES / "reference" / f"{name.replace(' ', '_')}.npz")
        steps = np.arange(len(z["train_acc"])) * 100
        ax.plot(steps, z["train_acc"], color=C_MEM, label="train")
        ax.plot(steps, z["test_acc"], color=C_GROK, label="held-out")
        ax.axhline(0.9, color=INK2, ls=":", lw=1)
        ax.set_title(f"{name}\nlr={s['lr']:g}, wd={s['wd']:.3g}", fontsize=10)
        ax.set_xlabel("update")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, loc="center right")
    fig.suptitle("Four phases at the paper's Fig. 7 cells (our rebuild)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_reference_cells.png", dpi=160)
    plt.close(fig)


def sweep_figures():
    folder = RES / "mlp23"
    try:
        S = load_sweep(folder)
    except FileNotFoundError:
        return
    phases = np.array([classify(a, b) for a, b in zip(S["train_acc"], S["test_acc"])])
    lr, wd = S["lr"], S["wd"]
    counts = {ph: int((phases == ph).sum()) for ph in (GROK, MEMORIZE, FORGET, NOFIT)}
    metrics["sweep"] = dict(n_runs=int(len(phases)), phase_counts=counts)

    # phase map on the paper's grid (cells we did not run stay blank)
    code = {GROK: 0, MEMORIZE: 1, FORGET: 2, NOFIT: 3}
    grid = np.full((90, 84), np.nan)
    for li, wi, ph in zip(S["lr_idx"], S["wd_idx"], phases):
        grid[wi, li] = code[ph]
    used_l, used_w = np.unique(S["lr_idx"]), np.unique(S["wd_idx"])
    sub = grid[np.ix_(used_w, used_l)]
    L, W = PAPER_LRS[used_l], PAPER_WDS[used_w]

    def edges(v):
        lv = np.log10(v)
        if len(lv) == 1:
            return 10 ** np.r_[lv[0] - 0.03, lv[0] + 0.03]
        mid = (lv[1:] + lv[:-1]) / 2
        return 10 ** np.r_[lv[0] - (mid[0] - lv[0]), mid, lv[-1] + (lv[-1] - mid[-1])]

    pts = boundary_points(lr, wd, phases)
    fit_mask = (pts[:, 0] >= 2e-5) & (pts[:, 0] <= 2e-4) if len(pts) else np.array([], bool)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cmap = ListedColormap([C_GROK, C_MEM, C_FORGET, C_NOFIT])
    ax.pcolormesh(edges(W), edges(L), sub.T, cmap=cmap, vmin=-0.5, vmax=3.5,
                  edgecolors=SURFACE, linewidth=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(False)
    wline = np.logspace(np.log10(5e-6), np.log10(2e-3), 100)
    ax.plot(wline, PAPER_A / wline, color=INK, ls="--", lw=1.6, label="paper fit  lr = 1.09e-4 / wd")
    if fit_mask.sum() >= 3:
        A, rmse = fit_inverse(pts[fit_mask, 0], pts[fit_mask, 1])
        k, c, r2 = fit_powerlaw(pts[fit_mask, 0], pts[fit_mask, 1])
        metrics["boundary"] = dict(A=A, log_rmse=rmse, free_slope=k, free_slope_r2=r2,
                                   n_points=int(fit_mask.sum()), paper_A=PAPER_A,
                                   paper_free_slope=PAPER_SLOPE)
        ax.plot(wline, A / wline, color=INK, lw=1.6, label=f"our fit  lr = {A:.3g} / wd")
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=SURFACE, edgecolor=INK, lw=1.2, zorder=3,
                   label="boundary midpoints")
    ax.set_ylim(L[0] / 1.1, L[-1] * 1.1); ax.set_xlim(W[0] / 1.1, W[-1] * 1.1)
    ax.set_xlabel("weight decay  λ")
    ax.set_ylabel("learning rate  η")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_GROK, C_MEM, C_FORGET, C_NOFIT)]
    leg1 = ax.legend(handles, ["grokking", "memorization", "forgetting", "no fitting"],
                     frameon=True, framealpha=0.95, loc="upper right", fontsize=9, title="phase")
    ax.add_artist(leg1)
    ax.legend(frameon=True, framealpha=0.95, loc="lower left", fontsize=9)
    ax.set_title(f"Phase map, mod-23 MLP, {len(phases):,} networks", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig3a_phase_map.png", dpi=160)
    plt.close(fig)

    # transition-time scaling: s10 vs 1/(lr*wd) for runs that grok
    g = phases == GROK
    s10 = np.array([first_crossing(t, 0.10) for t in S["test_acc"][g]], dtype=float)
    s90 = np.array([first_crossing(t, 0.90) for t in S["test_acc"][g]], dtype=float)
    x = 1.0 / (lr[g] * wd[g])
    ok = s10 > 0
    if ok.sum() >= 5:
        k, c, r2 = fit_powerlaw(x[ok], s10[ok])
        k90, c90, r290 = fit_powerlaw(x[s90 > 0], s90[s90 > 0])
        # Theory with the fitting offset: s10 = s_fit + C/(lr*wd), s_fit ~ alpha/lr
        #   =>  lr * s10 = alpha + C / wd   (linear in 1/wd)
        yy, xx = lr[g][ok] * s10[ok], 1.0 / wd[g][ok]
        C_off, alpha = np.polyfit(xx, yy, 1)
        r2_off = 1 - np.sum((yy - (alpha + C_off * xx)) ** 2) / np.sum((yy - yy.mean()) ** 2)
        metrics["timing_law_with_offset"] = dict(alpha=float(alpha), C=float(C_off), r2=float(r2_off),
                                                 note="lr*s10 = alpha + C/wd")
        metrics["timing_law"] = dict(
            n_runs=int(ok.sum()), slope_s10=k, r2_s10=r2, slope_s90=k90, r2_s90=r290,
            median_lr_wd_s10=float(np.median(lr[g][ok] * wd[g][ok] * s10[ok])),
            predicted_slope=1.0)
        fig, ax = plt.subplots(figsize=(6.4, 5))
        ax.scatter(x[ok], s10[ok], s=26, color=C_GROK, edgecolor=SURFACE, lw=0.8, label="grokking runs")
        xs = np.logspace(np.log10(x[ok].min()), np.log10(x[ok].max()), 50)
        ax.plot(xs, np.exp(c) * xs ** k, color=INK, lw=1.6, label=f"fit: slope {k:.2f}, R² {r2:.2f}")
        ax.plot(xs, np.median(s10[ok] / x[ok]) * xs, color=INK2, ls="--", lw=1.4,
                label="theory: slope 1 (not fitted)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("1 / (η λ)")
        ax.set_ylabel("first update with held-out acc ≥ 10%")
        ax.set_title("Grokking time scales as 1/(learning rate × weight decay)", loc="left", fontsize=11)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "fig3b_timing_law.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 5))
        ax.scatter(xx, yy, s=26, color=C_GROK, edgecolor=SURFACE, lw=0.8, label="grokking runs")
        xs = np.linspace(0, xx.max() * 1.05, 50)
        ax.plot(xs, alpha + C_off * xs, color=INK, lw=1.6,
                label=f"η·s₁₀ = {alpha:.0f} + {C_off:.2f}/λ   (R² {r2_off:.2f})")
        ax.set_xlabel("1 / λ")
        ax.set_ylabel("η × (first update with held-out acc ≥ 10%)")
        ax.set_title("Same data, with the fitting stage as an offset", loc="left", fontsize=11)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(FIG / "fig3b_timing_law_offset.png", dpi=160)
        plt.close(fig)


def ntk_figure():
    path = RES / "ntk97" / "label_kernels.npz"
    if not path.exists():
        return
    z = np.load(path)
    steps, Ks = z["steps"], z["kernels"]
    acc = np.loadtxt(RES / "ntk97" / "accuracy.csv", delimiter=",", skiprows=1)
    circ = [circulant_fraction(K) for K in Ks]
    amp = [circulant_amplitude(K) for K in Ks]
    peaks = [float(np.mean([fourier_peak(np.linalg.eigh(K)[1][:, -1 - i])[0] for i in range(4)])) for K in Ks]
    metrics["ntk97"] = dict(steps=steps.tolist(), circulant_fraction=circ, circulant_amplitude=amp,
                            top4_eigvec_fourier_peak_mean=peaks,
                            first_train_99=first_crossing(acc[:, 1], 0.99),
                            first_train_100=first_crossing(acc[:, 1], 0.999),
                            first_test_50=first_crossing(acc[:, 2], 0.5),
                            first_test_90=first_crossing(acc[:, 2], 0.9))
    raw = z["raw_kernels"] if "raw_kernels" in z.files else None
    if raw is not None:
        spec = np.array([circulant_spectrum(K) for K in raw])  # [T, p]
        top_k = int(np.argmax(spec[-1, 1:])) + 1
        # strongest non-constant Fourier mode, measured above the typical mode: the diagonal adds
        # the same constant to every frequency, so subtracting the median removes it exactly
        task = spec[:, 1:].max(1) - np.median(spec[:, 1:], axis=1)
        scale = np.array([np.mean(np.diag(K)) for K in raw])
        metrics["ntk97"].update(raw_top_fourier_eigenvalue=task.tolist(), raw_mean_diagonal=scale.tolist(),
                                top_frequency_at_end=top_k)
    show = [s for s in (0, 3000, 8000, 20000) if s in steps]
    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])
    for i, s in enumerate(show):
        ax = fig.add_subplot(gs[0, i])
        K = Ks[list(steps).index(s)].copy()
        np.fill_diagonal(K, np.nan)  # diagonal is 1 by construction; hide it to show structure
        K = K - np.nanmean(K)
        lim = np.nanmax(np.abs(K))
        ax.imshow(K, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"update {s:,}  (circulant share {circulant_fraction(Ks[list(steps).index(s)]):.2f})",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    t_fit = first_crossing(acc[:, 1], 0.999)
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(acc[:, 0], acc[:, 1], color=C_MEM, label="train acc")
    ax.plot(acc[:, 0], acc[:, 2], color=C_GROK, label="held-out acc")
    ax.plot(steps, circ, color=INK, marker="o", ms=6, lw=1.6, label="circulant share (off-diag)")
    ax.plot(steps, peaks, color=C_FORGET, marker="s", ms=6, lw=1.6, label="top-4 eigvec Fourier peak")
    if t_fit:
        ax.axvline(t_fit, color=INK2, ls=":", lw=1)
        ax.text(t_fit, 0.03, " train acc = 100%", color=INK2, fontsize=9)
    ax.set_xlabel("update"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Normalized structure (shape only)", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    if raw is not None:
        ax = fig.add_subplot(gs[1, 2:])
        ax.plot(steps, task, color=INK, marker="o", ms=6, lw=1.6,
                label=f"strongest Fourier mode above median (k={top_k} at end)")
        if t_fit:
            ax.axvline(t_fit, color=INK2, ls=":", lw=1)
        ax.set_xlabel("update"); ax.set_ylim(bottom=0)
        ax.set_title("Raw task-aligned NTK strength (proxy for the paper's Λ)", loc="left", fontsize=10)
        ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Label-space NTK of the mod-97 MLP develops Fourier (circulant) structure", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ntk_fourier.png", dpi=160)
    plt.close(fig)


def ridge_figure():
    path = RES / "ridge" / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    metrics["ridge"] = summary
    states = [("after_24000", "_24000", "after 24,000 updates"), ("after_300000", "_300000", "after 300,000 updates")]
    fig, axes = plt.subplots(len(summary), 2, figsize=(12, 4.4 * len(summary)), squeeze=False)
    for r, name in enumerate(summary):
        for col, (key, suffix, label) in enumerate(states):
            ax = axes[r, col]
            if key not in summary[name]:
                ax.set_visible(False)
                continue
            z = np.load(RES / "ridge" / f"{name}{suffix}.npz")
            keep = np.abs(z["yj"]) > 1e-3 * np.abs(z["yj"]).max()
            lam, ratio = z["evals"][keep], (z["sj"] / z["yj"])[keep]
            pos = (lam > 0) & (-ratio > 0)
            ax.scatter(lam[pos], -ratio[pos], s=6, color=C_GROK, alpha=0.35, lw=0, label="measured  −σⱼ / yⱼ")
            ls = np.logspace(np.log10(lam[pos].min()), np.log10(lam[pos].max()), 200)
            ax.plot(ls, z["c"] / (ls + z["c"]), color=INK, lw=1.8, label="Theorem 1:  c / (Λⱼ + c)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("NTK eigenvalue Λⱼ")
            ax.set_ylabel("share of target left as error")
            st = summary[name][key]
            ax.set_title(f"{name} cell, {label}\nrel. error {st['relative_error_vs_ridge_prediction']:.2g}, "
                         f"|∇obj|/|∇data| = {st['grad_norm_ratio']:.2g}", loc="left", fontsize=10)
            ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "thm1_ridge_residual.png", dpi=160)
    plt.close(fig)


reference_figure()
sweep_figures()
ntk_figure()
ridge_figure()
(RES / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({k: v for k, v in metrics.items() if k != "reference_cells"}, indent=2)[:3000])
