import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# 配置全局样式
STYLE = {
    "ground": "#DCECF4",
    "module": "#F6E8B1",
    "flow":   "#CFE0C7",
    "fusion": "#E0E0E0",
    "output": "#CFE0C7",
    "white":  "#FFFFFF",
    "edge":   "#333333",
    "font":   "sans-serif"
}

def draw_node(ax, rect, text, fc=STYLE["white"], fontsize=11):
    """绘制带文本的圆角矩形节点"""
    x, y, w, h = rect
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=1.5, edgecolor=STYLE["edge"], facecolor=fc,
        zorder=2
    )
    ax.add_patch(box)
    
    # 自动处理换行
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            linespacing=1.2, zorder=3)
    return (x, y, w, h)

def draw_arrow(ax, start_rect, end_rect, start_side="bottom", end_side="top"):
    """
    基于矩形位置自动计算箭头的连接点
    sides: 'top', 'bottom', 'left', 'right'
    """
    def get_point(r, side):
        x, y, w, h = r
        if side == "top":    return (x + w/2, y + h)
        if side == "bottom": return (x + w/2, y)
        if side == "left":   return (x, y + h/2)
        if side == "right":  return (x + w, y + h/2)
        return (x + w/2, y + h/2)

    p1 = get_point(start_rect, start_side)
    p2 = get_point(end_rect, end_side)
    
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=STYLE["edge"], 
                shrinkA=2, shrinkB=2, mutation_scale=15),
                zorder=1)

def main():
    fig, ax = plt.subplots(figsize=(11, 7), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 1. 定义节点坐标 (x, y, w, h)
    # 输入层
    n_traj = draw_node(ax, (0.05, 0.88, 0.4, 0.08), "Cyclist Trajectories\n(pre-computed tracks)")
    n_scene = draw_node(ax, (0.55, 0.88, 0.4, 0.08), "Scene Spatial Primitives\n(ROIs: sidewalk, bike lane, roadway)")

    # 处理层
    n_ground = draw_node(ax, (0.15, 0.73, 0.7, 0.09), "Scene-Aware Spatial Grounding\n(ROI-based semantic labeling)", fc=STYLE["ground"])
    n_infer  = draw_node(ax, (0.2, 0.61, 0.6, 0.07), "Parallel Behavior Inference\n(rule-based, scene-aware)")

    # 并行分析模块
    n_cross = draw_node(ax, (0.05, 0.44, 0.25, 0.1), "Crossing\nAnalysis", fc=STYLE["module"])
    n_space = draw_node(ax, (0.375, 0.44, 0.25, 0.1), "Space Usage\nAnalysis", fc=STYLE["module"])
    n_flow  = draw_node(ax, (0.7, 0.43, 0.25, 0.12), "Flow Alignment Analysis\n\n• Right-Side Consistency\n• Wrong-Way Detection", fc=STYLE["flow"], fontsize=9)

    # 融合与输出
    n_fusion = draw_node(ax, (0.25, 0.28, 0.5, 0.07), "Event-Level Behavior Fusion", fc=STYLE["fusion"])
    n_output = draw_node(ax, (0.2, 0.12, 0.6, 0.08), "Location-Level Behavior Metrics\n(Dominant space, occupancy, wrong-way)", fc=STYLE["output"])

    # 2. 建立连接 (逻辑更清晰)
    # 输入 -> Grounding (斜线)
    draw_arrow(ax, n_traj, n_ground, "bottom", "top")
    draw_arrow(ax, n_scene, n_ground, "bottom", "top")
    
    # Grounding -> Inference
    draw_arrow(ax, n_ground, n_infer)

    # Inference -> 三个并行模块
    draw_arrow(ax, n_infer, n_cross)
    draw_arrow(ax, n_infer, n_space)
    draw_arrow(ax, n_infer, n_flow)

    # 并行模块 -> Fusion
    draw_arrow(ax, n_cross, n_fusion)
    draw_arrow(ax, n_space, n_fusion)
    draw_arrow(ax, n_flow, n_fusion)

    # Fusion -> Output
    draw_arrow(ax, n_fusion, n_output)

    # 保存
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/pipeline_optimized.png", bbox_inches="tight", bg_color="white")
    print("Successfully saved optimized pipeline.")

if __name__ == "__main__":
    main()