# make_pipeline_buildsys_spaced.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams
import os

def add_box(ax, x, y, w, h, text, fc="white", ec="black",
            lw=1.5, fontsize=14, fontweight="bold", radius=0.02):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)
    return box

def add_arrow(ax, x1, y1, x2, y2, lw=1.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color="black",
                        mutation_scale=14)
    )

def main():
    rcParams["figure.dpi"] = 150
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- Colors (print-friendly) ----
    c_ground = "#dcecf4"
    c_module = "#f6e8b1"
    c_flow   = "#cfe0c7"
    c_fusion = "#e0e0e0"
    c_out    = "#cfe0c7"

    # ---- Top inputs (拉开横向间距) ----
    b_traj = add_box(
        ax, 0.06, 0.86, 0.36, 0.11,
        "Cyclist Trajectories\n(pre-computed tracks)",
        fontsize=16
    )
    b_roi = add_box(
        ax, 0.58, 0.86, 0.36, 0.11,
        "Scene Spatial Primitives\n(ROIs: sidewalk, bike lane, roadway)",
        fontsize=16
    )

    # ---- Spatial Grounding（纵向拉开）----
    b_ground = add_box(
        ax, 0.20, 0.68, 0.60, 0.11,
        "Scene-Aware Spatial Grounding\n(ROI-based semantic labeling)",
        fc=c_ground, fontsize=17
    )

    # ---- Parallel inference label ----
    b_parallel = add_box(
        ax, 0.22, 0.54, 0.56, 0.10,
        "Parallel Behavior Inference\nParallel rule-based behavior inference modules",
        fontsize=15
    )

    # ---- Scene-aware processing container（更低一点）----
    container = FancyBboxPatch(
        (0.06, 0.24), 0.88, 0.26,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.5, edgecolor="black", facecolor="white"
    )
    ax.add_patch(container)
    ax.text(0.50, 0.48, "Scene-Aware Processing",
            ha="center", va="center",
            fontsize=16, fontweight="bold")

    # ---- Parallel modules（横向明显拉开）----
    b_cross = add_box(
        ax, 0.10, 0.30, 0.22, 0.14,
        "Crossing\nAnalysis",
        fc=c_module, fontsize=16
    )
    b_space = add_box(
        ax, 0.39, 0.30, 0.22, 0.14,
        "Space Usage\nAnalysis",
        fc=c_module, fontsize=16
    )
    b_flow = add_box(
        ax, 0.68, 0.285, 0.24, 0.17,
        "Flow Alignment Analysis\n\n• Right-Side Consistency\n• Wrong-Way Detection",
        fc=c_flow, fontsize=13
    )

    # ---- Fusion & Output（纵向拉开）----
    b_fusion = add_box(
        ax, 0.28, 0.13, 0.44, 0.08,
        "Event-Level Behavior Fusion",
        fc=c_fusion, fontsize=15
    )
    b_out = add_box(
        ax, 0.26, 0.02, 0.48, 0.08,
        "Location-Level Behavior Metrics\n(Dominant space, occupancy, wrong-way)",
        fc=c_out, fontsize=15
    )

    # ---- Arrows ----
    add_arrow(ax, 0.24, 0.86, 0.38, 0.79)
    add_arrow(ax, 0.76, 0.86, 0.62, 0.79)

    add_arrow(ax, 0.50, 0.68, 0.50, 0.64)
    add_arrow(ax, 0.50, 0.54, 0.50, 0.50)

    add_arrow(ax, 0.28, 0.54, 0.21, 0.44)
    add_arrow(ax, 0.50, 0.54, 0.50, 0.44)
    add_arrow(ax, 0.72, 0.54, 0.80, 0.44)

    add_arrow(ax, 0.21, 0.30, 0.40, 0.21)
    add_arrow(ax, 0.50, 0.30, 0.50, 0.21)
    add_arrow(ax, 0.80, 0.285, 0.60, 0.21)

    add_arrow(ax, 0.50, 0.13, 0.50, 0.10)

    # ---- Save ----
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "pipeline_buildsys_spaced.png"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "pipeline_buildsys_spaced.pdf"),
                bbox_inches="tight")

    print("Saved: figures/pipeline_buildsys_spaced.png and .pdf")

if __name__ == "__main__":
    main()
