#!/usr/bin/env python3
"""Regenerate the two data figures for main_v3.tex (TRB 2027 draft).

Reads data/manual_counts_new.csv (frozen manual benchmark) plus the frozen
pipeline counts. Outputs paper/fig_validation.png and paper/fig_scatter_v3.png.
Prints verification stats that must match the frozen benchmark numbers:
1,951 manual / 1,823 pipe events, Spearman rho = 0.88 (ours) / 0.79 (TBAG).

Palette (validated, light surface): blue #2a78d6, green #1baf7a. Green fails
3:1 contrast on white, so every green mark also carries a distinct shape and
direct labels where identity matters.
"""
import csv
import os

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "manual_counts_new.csv")

BLUE = "#2a78d6"   # this study / manual counts
GREEN = "#1baf7a"  # TBAG / pipeline counts
INK = "#33322e"
MUTED = "#6b6a66"

# Frozen pipeline per-deployment counts (outputs_new, commit 74a9923).
PIPE = {"02": 4, "04": 220, "04-2": 146, "06": 74, "08": 183, "10": 143,
        "11": 58, "12": 89, "13": 62, "15": 91, "16": 61, "17": 458,
        "18": 17, "19": 55, "19-2": 31, "21": 10, "23": 91, "24": 27,
        "25": 3}

# TBAG 2024 volunteer counts, 16 intersections (LocationID order
# 102,106,113,114,115,119,120,121,126,131,133,135,139,157,160,172).
TBAG_SW = [48, 19, 31, 11, 66, 50, 35, 54, 70, 71, 17, 60, 8, 48, 40, 1]
TBAG_WW = [21, 9, 16, 12, 38, 25, 9, 11, 32, 22, 10, 21, 2, 25, 3, 1]

# Excluded from the correlation (n<10 manual events): 02, 21, 25.
SCATTER_EXCLUDE = {"02", "21", "25"}


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def load_manual():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k == "loc" else float(v)) for k, v in r.items()})
    return rows


def fig_validation(rows, out):
    locs = [r["loc"] for r in rows]
    manual = [r["total"] for r in rows]
    pipe = [PIPE[l] for l in locs]
    ww_rate = [r["ww_total"] / r["total"] for r in rows]
    ci = [wilson_ci(r["ww_total"], r["total"]) for r in rows]

    y = np.arange(len(locs))[::-1]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.0, 5.6), dpi=300, sharey=False,
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.14})

    h = 0.38
    ax1.barh(y + h / 2 + 0.01, manual, height=h, color=BLUE, label="Manual")
    ax1.barh(y - h / 2 - 0.01, pipe, height=h, color=GREEN, label="Pipeline")
    for yi, m, p in zip(y, manual, pipe):
        ax1.text(m + 5, yi + h / 2 + 0.01, f"{int(m)}", va="center",
                 fontsize=6.2, color=INK)
        ax1.text(p + 5, yi - h / 2 - 0.01, f"{int(p)}", va="center",
                 fontsize=6.2, color=MUTED)
    ax1.set_yticks(y, [f"Loc {l}" for l in locs], fontsize=7.5)
    ax1.set_xlabel("Cyclist events", fontsize=8.5)
    ax1.set_title("(a) Event counts: manual vs. pipeline", fontsize=9, loc="left")
    ax1.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax1.set_xlim(0, max(manual) * 1.14)

    for yi, r, (lo, hi) in zip(y, ww_rate, ci):
        ax2.plot([lo * 100, hi * 100], [yi, yi], color=MUTED, lw=1.4, zorder=2)
    ax2.scatter([r * 100 for r in ww_rate], y, marker="D", s=26, color=BLUE,
                zorder=3, label="Manual WW rate")
    for yi, r in zip(y, ww_rate):
        ax2.text(r * 100, yi + 0.32, f"{r * 100:.0f}", ha="center",
                 fontsize=6.2, color=INK)
    ax2.set_yticks(y, ["" for _ in locs])
    ax2.set_xlabel("Wrong-way share of events (%)", fontsize=8.5)
    ax2.set_title("(b) Wrong-way rate (Wilson 95% CI)", fontsize=9, loc="left")
    ax2.set_xlim(-2, 100)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e5e1", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    n_man, n_pipe = sum(manual), sum(pipe)
    print(f"validation: manual {int(n_man)} / pipe {int(n_pipe)} "
          f"(expect 1951 / 1823), recall {n_pipe / n_man:.3f}")
    r = stats.pearsonr(manual, pipe)
    print(f"count Pearson r = {r.statistic:.3f} (expect 0.991)")


def fig_scatter(rows, out):
    sub = [r for r in rows if r["loc"] not in SCATTER_EXCLUDE]
    sw = [100 * (r["fwd_sw"] + r["ww_sw"]) / r["total"] for r in sub]
    ww = [100 * r["ww_total"] / r["total"] for r in sub]
    rho_us = stats.spearmanr(sw, ww)
    rho_tb = stats.spearmanr(TBAG_SW, TBAG_WW)
    print(f"ours   Spearman rho = {rho_us.statistic:.3f} p = {rho_us.pvalue:.1e}"
          f" n = {len(sw)} (expect 0.88, 6.4e-06, 16)")
    print(f"TBAG   Spearman rho = {rho_tb.statistic:.3f} p = {rho_tb.pvalue:.1e}"
          f" (expect 0.788, 3e-04)")

    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=300)
    ax.scatter(sw, ww, s=42, color=BLUE, marker="o", zorder=3,
               label=rf"This study (n={len(sw)}), $\rho$ = {rho_us.statistic:.2f}")
    ax.scatter(TBAG_SW, TBAG_WW, s=42, color=GREEN, marker="D", zorder=3,
               label=rf"TBAG 2024 (n=16), $\rho$ = {rho_tb.statistic:.2f}")
    ax.set_xlabel("Sidewalk share of cyclist events (%)", fontsize=9)
    ax.set_ylabel("Wrong-way share of events (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e1", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.8, frameon=False, loc="lower right")
    ax.set_xlim(-3, 103)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rows = load_manual()
    fig_validation(rows, os.path.join(HERE, "fig_validation.png"))
    fig_scatter(rows, os.path.join(HERE, "fig_scatter_v3.png"))
    print("wrote fig_validation.png, fig_scatter_v3.png")
