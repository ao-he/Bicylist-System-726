import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams
import os

def add_box(ax, x, y, w, h, text, fc="white",
            fontsize=15, lw=1.5, radius=0.02):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw, edgecolor="black", facecolor=fc
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold")
    return box

def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black",
                        mutation_scale=14)
    )

def main():
    rcParams["figure.dpi"] = 150
    fig, ax = plt.subplots(figsize=(12.5, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Colors
    c_ground = "#dcecf4"
    c_module = "#f6e8b1"
    c_flow   = "#cfe0c7"
    c_fusion = "#e0e0e0"
    c_out    = "#cfe0c7"

    # ---- Inputs ----
    add_box(ax, 0.08, 0.87, 0.36, 0.10,
            "Cyclist Trajectories\n(pre-computed tracks)", fontsize=16)
    add_box(ax, 0.56, 0.87, 0.36, 0.10,
            "Scene Spatial Primitives\n(ROIs: sidewalk, bike lane, roadway)",
            fontsize=16)

    # ---- Grounding ----
    add_box(ax, 0.18, 0.72, 0.64, 0.10,
            "Scene-Aware Spatial Grounding\n(ROI-based semantic labeling)",
            fc=c_ground, fontsize=17)

    # arrows to grounding
    arrow(ax, 0.26, 0.87, 0.40, 0.82)
    arrow(ax, 0.74, 0.87, 0.60, 0.82)

    # ---- Parallel inference header ----
    add_box(ax, 0.22, 0.60, 0.56, 0.08,
            "Parallel Behavior Inference\n(rule-based, scene-aware)",
            fontsize=15)

    arrow(ax, 0.50, 0.72, 0.50, 0.68)

    # ---- Parallel modules (clean horizontal layout) ----
    add_box(ax, 0.08, 0.42, 0.24, 0.12,
            "Crossing\nAnalysis", fc=c_module, fontsize=16)
    add_box(ax, 0.38, 0.42, 0.24, 0.12,
            "Space Usage\nAnalysis", fc=c_module, fontsize=16)
    add_box(ax, 0.68, 0.41, 0.24, 0.14,
            "Flow Alignment Analysis\n\n• Right-Side Consistency\n• Wrong-Way Detection",
            fc=c_flow, fontsize=13)

    arrow(ax, 0.20, 0.60, 0.20, 0.54)
    arrow(ax, 0.50, 0.60, 0.50, 0.54)
    arrow(ax, 0.80, 0.60, 0.80, 0.55)

    # ---- Fusion ----
    add_box(ax, 0.28, 0.25, 0.44, 0.08,
            "Event-Level Behavior Fusion",
            fc=c_fusion, fontsize=15)

    arrow(ax, 0.20, 0.42, 0.40, 0.33)
    arrow(ax, 0.50, 0.42, 0.50, 0.33)
    arrow(ax, 0.80, 0.41, 0.60, 0.33)

    # ---- Output ----
    add_box(ax, 0.26, 0.12, 0.48, 0.08,
            "Location-Level Behavior Metrics\n(Dominant space, occupancy, wrong-way)",
            fc=c_out, fontsize=15)

    arrow(ax, 0.50, 0.25, 0.50, 0.20)

    # Save
    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "pipeline_buildsys_final.png"),
                bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "pipeline_buildsys_final.pdf"),
                bbox_inches="tight")

    print("Saved: figures/pipeline_buildsys_final.png / .pdf")

if __name__ == "__main__":
    main()
