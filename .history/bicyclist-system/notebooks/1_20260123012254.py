import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# --- 样式配置 ---
STYLE = {
    "ground": "#DCECF4",  # 淡蓝色
    "module": "#F6E8B1",  # 淡黄色
    "flow":   "#CFE0C7",  # 淡绿色
    "fusion": "#E0E0E0",  # 灰色
    "output": "#CFE0C7",  # 淡绿色
    "white":  "#FFFFFF",
    "edge":   "#333333",
}

def draw_node(ax, rect, text, fc=STYLE["white"], fontsize=11):
    """绘制带文本和阴影效果的圆角矩形节点"""
    x, y, w, h = rect
    
    # 增加微小阴影效果 (简单实现：在底层画一个偏移的灰色框)
    shadow_offset = 0.003
    ax.add_patch(FancyBboxPatch(
        (x + shadow_offset, y - shadow_offset), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor="#BCBCBC", alpha=0.5, zorder=1
    ))

    # 主体框
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=1.2, edgecolor=STYLE["edge"], facecolor=fc,
        zorder=2
    )
    ax.add_patch(box)
    
    # 文本居中
    ax.text(x + w/2, y + h/2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            linespacing=1.2, zorder=3)
    return (x, y, w, h)

def draw_arrow(ax, start_rect, end_rect):
    """
    自动计算从上层节点底部中心到下层节点顶部中心的箭头
    """
    x1, y1, w1, h1 = start_rect
    x2, y2, w2, h2 = end_rect
    
    # 起点：上层方框底部中心；终点：下层方框顶部中心
    p1 = (x1 + w1/2, y1)
    p2 = (x2 + w2/2, y2 + h2)
    
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=STYLE["edge"], 
                shrinkA=2, shrinkB=2, mutation_scale=15),
                zorder=1)

def main():
    # 设置画布
    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- 1. 绘制节点 (基于你的 Pipeline 逻辑) --- [cite: 5, 7, 11]
    
    # 输入层 [cite: 1, 3]
    n_traj  = draw_node(ax, (0.05, 0.88, 0.4, 0.08), "Cyclist Trajectories\n(pre-computed tracks)")
    n_scene = draw_node(ax, (0.55, 0.88, 0.4, 0.08), "Scene Spatial Primitives\n(ROIs: sidewalk, bike lane, roadway)")

    # 空间接地层 [cite: 5]
    n_ground = draw_node(ax, (0.15, 0.73, 0.7, 0.09), "Scene-Aware Spatial Grounding\n(ROI-based semantic labeling)", fc=STYLE["ground"])
    
    # 推理层 [cite: 7]
    n_infer  = draw_node(ax, (0.2, 0.62, 0.6, 0.07), "Parallel Behavior Inference\n(rule-based, scene-aware)")

    # 并行分析模块 (Crossing, Space, Flow) [cite: 2, 8, 9]
    n_cross = draw_node(ax, (0.05, 0.44, 0.25, 0.11), "Crossing\nAnalysis", fc=STYLE["module"])
    n_space = draw_node(ax, (0.375, 0.44, 0.25, 0.11), "Space Usage\nAnalysis", fc=STYLE["module"])
    n_flow  = draw_node(ax, (0.7, 0.43, 0.25, 0.13), "Flow Alignment Analysis\n\n• Right-Side Consistency\n• Wrong-Way Detection", fc=STYLE["flow"], fontsize=9)

    # 融合与输出层 [cite: 11, 12]
    n_fusion = draw_node(ax, (0.25, 0.28, 0.5, 0.07), "Event-Level Behavior Fusion", fc=STYLE["fusion"])
    n_output = draw_node(ax, (0.2, 0.12, 0.6, 0.08), "Location-Level Behavior Metrics\n(Dominant space, occupancy, wrong-way)", fc=STYLE["output"])

    # --- 2. 建立连接箭头 ---
    
    # 输入 -> Grounding
    draw_arrow(ax, n_traj, n_ground)
    draw_arrow(ax, n_scene, n_ground)
    
    # Grounding -> Inference
    draw_arrow(ax, n_ground, n_infer)

    # Inference -> 三个并行子模块
    draw_arrow(ax, n_infer, n_cross)
    draw_arrow(ax, n_infer, n_space)
    draw_arrow(ax, n_infer, n_flow)

    # 并行模块 -> Fusion
    draw_arrow(ax, n_cross, n_fusion)
    draw_arrow(ax, n_space, n_fusion)
    draw_arrow(ax, n_flow, n_fusion)

    # Fusion -> Output
    draw_arrow(ax, n_fusion, n_output)

    # --- 3. 保存 ---
    os.makedirs("figures", exist_ok=True)
    # 修复：使用 facecolor="white" 替代错误的 bg_color
    plt.savefig("figures/pipeline_optimized.png", bbox_inches="tight", facecolor="white")
    plt.savefig("figures/pipeline_optimized.pdf", bbox_inches="tight", facecolor="white")
    
    plt.show()
    print("图像已成功保存至 figures 文件夹。")

if __name__ == "__main__":
    main()