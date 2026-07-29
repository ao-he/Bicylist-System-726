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
    label_off = {"02": (5, 4), "21": (5, 4), "25": (5, -9), "06": (5, -9),
                 "18": (-28, 4), "19": (5, -9), "19-2": (-40, 4)}
    for l, m, p in zip(locs, manual, pipe):
        ratio = p / m
        if ratio < 0.8 or ratio > 1.25:
            dx, dy = label_off.get(l, (5, 4))
            ax1.annotate(f"Loc {l}", (m, p), textcoords="offset points",
                         xytext=(dx, dy), fontsize=6.8, color=INK)
    ax1.text(0.05, 0.93, "$r$ = 0.99", transform=ax1.transAxes, fontsize=9)
    ax1.text(0.97, 0.03, "below line: pipeline undercounts",
             transform=ax1.transAxes, fontsize=6.8, ha="right", color=MUTED)
    ax1.set_xlabel("Manual count (events)", fontsize=8.5)
    ax1.set_ylabel("Pipeline count (events)", fontsize=8.5)
    ax1.set_title("(a) Event counts per deployment", fontsize=9, loc="left")

    # (b) horizontal: manual wrong-way rate per deployment (all 19, with
    # value labels), pooled manual-vs-pipeline pair at the bottom.
    per = sorted(rows, key=lambda r: r["ww_total"] / r["total"])
    ys_b = np.arange(len(per)) + 2.6
    vals = [100 * r["ww_total"] / r["total"] for r in per]
    ax2.barh(ys_b, vals, height=0.62, color=BLUE, zorder=3,
             label="Manual, per deployment")
    for y, r, v in zip(ys_b, per, vals):
        lo, hi = wilson_ci(r["ww_total"], r["total"])
        ax2.plot([lo * 100, hi * 100], [y, y], color="#4a4944", lw=0.9,
                 alpha=0.75, zorder=4)
        ax2.text(hi * 100 + 1.2, y, f"{v:.0f}%", va="center", fontsize=6.2,
                 color=MUTED)
    ax2.axhline(1.75, color="#c9c8c4", lw=0.8, zorder=2)
    m_lo, m_hi = wilson_ci(426, 1951)
    p_lo, p_hi = wilson_ci(49, 268)
    ax2.barh(1.05, 21.8, height=0.62, color=BLUE, zorder=3)
    ax2.barh(0.25, 18.3, height=0.62, color=GREEN, zorder=3,
             label="Pipeline, direction-known")
    ax2.plot([m_lo * 100, m_hi * 100], [1.05, 1.05], color="#4a4944",
             lw=0.9, zorder=4)
    ax2.plot([p_lo * 100, p_hi * 100], [0.25, 0.25], color="#4a4944",
             lw=0.9, zorder=4)
    ax2.text(m_hi * 100 + 1.2, 1.05, "21.8%", va="center", fontsize=6.8,
             fontweight="bold", color=INK)
    ax2.text(p_hi * 100 + 1.2, 0.25, "18.3%", va="center", fontsize=6.8,
             fontweight="bold", color=INK)
    ax2.set_yticks(list(ys_b) + [1.05, 0.25],
                   [f"Loc {r['loc']}" for r in per] + ["Pooled", ""],
                   fontsize=6.4)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-0.55, len(per) + 3.0)
    ax2.set_xlabel("Wrong-way share of events (%)", fontsize=8.5)
    ax2.legend(fontsize=6.8, frameon=False, loc="lower right")
    ax2.text(0.985, 0.30, "whiskers: Wilson 95% CI",
             transform=ax2.transAxes, fontsize=6.4, color=MUTED, ha="right")
    ax2.set_title("(b) Wrong-way rate by deployment and pooled",
                  fontsize=9, loc="left")

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(color="#e6e5e1", lw=0.6)
    ax1.set_aspect("equal")
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
