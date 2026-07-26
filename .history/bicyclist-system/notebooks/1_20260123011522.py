# make_pipeline_fig.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import rcParams

def add_box(ax, x, y, w, h, text, fc="white", ec="black", lw=1.5,
            fontsize=14, fontweight="bold", align="center", radius=0.02):
    """
    Add a rounded rectangle box with centered text.
    (x, y) is lower-left in axes fraction coordinates [0,1].
    """
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(box)

    ax.text(
        x + w/2, y + h/2, text,
        ha=align, va="center",
        fontsize=fontsize, fontweight=fontweight, family="DejaVu Sans"
    )
    return box

def add_arrow(ax, x1, y1, x2, y2, lw=1.3):
    """Arrow from (x1,y1) to (x2,y2) in axes fraction coordinates."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color="black",
                        shrinkA=0, shrinkB=0, mutation_scale=14)
    )

def main():
    # --- Figure setup ---
    rcParams["figure.dpi"] = 150
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Colors (subtle, print-friendly) ---
    c_ground = "#dcecf4"   # light blue
    c_module = "#f6e8b1"   # light yellow
    c_flow   = "#cfe0c7"   # light green
    c_fusion = "#d9d9d9"   # light gray
    c_out    = "#cfe0c7"   # light green (same as flow)

    # --- Boxes (positions in axes fraction) ---
    # Top inputs
    b_traj = add_box(
        ax, 0.10, 0.83, 0.34, 0.13,
        "Cyclist Trajectories\n(pre-computed tracks)",
        fc="white", fontsize=16
    )
    b_roi = add_box(
        ax, 0.56, 0.83, 0.34, 0.13,
        "Scene Spatial Primitives\n(ROIs: sidewalk, bike lane, roadway)",
        fc="white", fontsize=16
    )

    # Spatial grounding
    b_ground = add_box(
        ax, 0.22, 0.64, 0.56, 0.12,
        "Scene-Aware Spatial Grounding\n(ROI-based semantic labeling)",
        fc=c_ground, fontsize=17
    )

    # Parallel layer label
    b_parallel = add_box(
        ax, 0.22, 0.49, 0.56, 0.10,
        "Parallel Behavior Inference\nParallel rule-based behavior inference modules",
        fc="white", fontsize=15
    )

    # Big container for modules
    # (This is optional; helps with BuildSys-style grouping)
    container = FancyBboxPatch(
        (0.10, 0.21), 0.80, 0.25,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.5, edgecolor="black", facecolor="white"
    )
    ax.add_patch(container)
    ax.text(0.50, 0.44, "Scene-Aware Processing",
            ha="center", va="center", fontsize=16, fontweight="bold")

    # Modules inside container
    b_cross = add_box(
        ax, 0.14, 0.26, 0.22, 0.14,
        "Crossing\nAnalysis",
        fc=c_module, fontsize=16
    )
    b_space = add_box(
        ax, 0.39, 0.26, 0.22, 0.14,
        "Space Usage\nAnalysis",
        fc=c_module, fontsize=16
    )
    b_flow = add_box(
        ax, 0.64, 0.245, 0.24, 0.17,
        "Flow Alignment Analysis\n\n• Right-Side Consistency\n• Wrong-Way Detection",
        fc=c_flow, fontsize=13, fontweight="bold"
    )

    # Fusion and output
    b_fusion = add_box(
        ax, 0.28, 0.12, 0.44, 0.08,
        "Event-Level Behavior Fusion",
        fc=c_fusion, fontsize=15
    )
    b_out = add_box(
        ax, 0.27, 0.02, 0.46, 0.08,
        "Location-Level Behavior Metrics\n(Dominant space, occupancy, wrong-way)",
        fc=c_out, fontsize=15
    )

    # --- Arrows ---
    # From inputs to grounding
    add_arrow(ax, 0.27, 0.83, 0.38, 0.76)  # traj -> grounding (approx)
    add_arrow(ax, 0.73, 0.83, 0.62, 0.76)  # roi -> grounding (approx)

    # Grounding -> parallel label
    add_arrow(ax, 0.50, 0.64, 0.50, 0.59)

    # Parallel label -> container top
    add_arrow(ax, 0.50, 0.49, 0.50, 0.46)

    # Parallel label -> three modules (fan out)
    add_arrow(ax, 0.40, 0.49, 0.25, 0.40)  # to crossing
    add_arrow(ax, 0.50, 0.49, 0.50, 0.40)  # to space
    add_arrow(ax, 0.60, 0.49, 0.76, 0.40)  # to flow

    # Modules -> fusion
    add_arrow(ax, 0.25, 0.26, 0.40, 0.20)  # crossing -> fusion
    add_arrow(ax, 0.50, 0.26, 0.50, 0.20)  # space -> fusion
    add_arrow(ax, 0.76, 0.245, 0.60, 0.20) # flow -> fusion

    # Fusion -> output
    add_arrow(ax, 0.50, 0.12, 0.50, 0.10)

    # --- Save ---
    fig.tight_layout()
    fig.savefig("pipeline_buildsys.png", bbox_inches="tight")
    fig.savefig("pipeline_buildsys.pdf", bbox_inches="tight")
    print("Saved: pipeline_buildsys.png and pipeline_buildsys.pdf")

if __name__ == "__main__":
    main()
