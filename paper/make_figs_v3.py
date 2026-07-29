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

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

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

# Frozen pipeline direction results (against, dir_known) per deployment.
PIPE_WW = {"02": (0, 1), "04": (11, 35), "04-2": (6, 16), "06": (0, 2),
           "08": (6, 9), "10": (1, 4), "11": (1, 2), "12": (2, 11),
           "13": (1, 8), "15": (2, 11), "16": (5, 11), "17": (5, 115),
           "18": (0, 0), "19": (2, 15), "19-2": (1, 3), "21": (0, 2),
           "23": (6, 21), "24": (0, 2), "25": (0, 0)}

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.7, 3.35), dpi=300,
                                   gridspec_kw={"wspace": 0.30})

    # (a) counts agreement: pipeline vs manual, identity line, log-log
    lim = (1.5, 700)
    ax1.plot(lim, lim, ls=(0, (4, 3)), color="#8a8884", lw=1.0, zorder=1)
    ax1.scatter(manual, pipe, s=30, facecolor=BLUE, alpha=0.8,
                edgecolor="#1b4e8f", linewidth=0.8, zorder=3)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ticks = [2, 5, 10, 20, 50, 100, 200, 500]
    ax1.set_xticks(ticks, [str(t) for t in ticks])
    ax1.set_yticks(ticks, [str(t) for t in ticks])
    ax1.minorticks_off()
    for l, m, p in zip(locs, manual, pipe):
        if l in ("17", "21", "02", "06"):
            ax1.annotate(f"Loc {l}", (m, p), textcoords="offset points",
                         xytext=(5, -9 if l in ("06",) else 5), fontsize=6.8,
                         color=INK)
    ax1.text(0.05, 0.93, "$r$ = 0.99", transform=ax1.transAxes, fontsize=9)
    ax1.text(0.97, 0.03, "below line: pipeline undercounts",
             transform=ax1.transAxes, fontsize=6.8, ha="right", color=MUTED)
    ax1.set_xlabel("Manual count (events)", fontsize=8.5)
    ax1.set_ylabel("Pipeline count (events)", fontsize=8.5)
    ax1.set_title("(a) Event counts per deployment", fontsize=9, loc="left")

    # (b) wrong-way agreement: pipeline (dir-known, n>=8) vs manual
    ax2.plot([0, 100], [0, 100], ls=(0, (4, 3)), color="#8a8884", lw=1.0,
             zorder=1)
    for r in rows:
        a, n = PIPE_WW[r["loc"]]
        if n < 8:
            continue
        mx = 100 * r["ww_total"] / r["total"]
        py = 100 * a / n
        lo, hi = wilson_ci(a, n)
        ax2.plot([mx, mx], [lo * 100, hi * 100], color=BLUE, lw=1.1,
                 alpha=0.45, zorder=2)
        ax2.scatter([mx], [py], s=30, facecolor=BLUE, alpha=0.85,
                    edgecolor="#1b4e8f", linewidth=0.8, zorder=3)
        if r["loc"] in ("08", "17"):
            ax2.annotate(f"Loc {r['loc']}", (mx, py),
                         textcoords="offset points", xytext=(5, 3),
                         fontsize=6.8, color=INK)
    ax2.scatter([21.8], [18.3], marker="s", s=42, facecolor=GREEN,
                edgecolor="#0e7a54", linewidth=0.9, zorder=4)
    ax2.annotate("pooled\n21.8 vs 18.3", (21.8, 18.3),
                 textcoords="offset points", xytext=(7, -16), fontsize=6.8,
                 color=INK)
    ax2.set_xlim(0, 75); ax2.set_ylim(0, 75)
    ax2.set_xlabel("Manual wrong-way rate, all events (%)", fontsize=8.5)
    ax2.set_ylabel("Pipeline wrong-way rate,\ndirection-known (%)", fontsize=8.5)
    ax2.set_title("(b) Wrong-way rate per deployment", fontsize=9, loc="left")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(color="#e6e5e1", lw=0.6)
        ax.set_aspect("equal")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    n_man, n_pipe = sum(manual), sum(pipe)
    print(f"validation: manual {int(n_man)} / pipe {int(n_pipe)} "
          f"(expect 1951 / 1823), recall {n_pipe / n_man:.3f}")
    r = stats.pearsonr(manual, pipe)
    pa = sum(v[0] for v in PIPE_WW.values())
    pn = sum(v[1] for v in PIPE_WW.values())
    print(f"pipeline direction pooled: {pa}/{pn} = {pa/pn:.3f} "
          f"(expect 49/268 = 0.183)")
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
