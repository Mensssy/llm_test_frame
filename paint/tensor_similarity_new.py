from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import matplotlib.patches as mpatches

# =========================
# 全局绘图风格
# =========================
COLORS = {
    "Origin": "#fed070",  # 浅蓝
    "Delta": "#abdce0",  # 浅黄abdce0
}


def set_academic_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# =========================
# 数据计算
# =========================
def convert_to_numpy(t: torch.Tensor) -> np.ndarray:
    if t.is_cuda:
        t = t.cpu()
    if t.requires_grad:
        t = t.detach()
    return t.numpy().astype(np.float32)


def compute_origin_delta(
    tensors: dict,
    input_id: int,
    model_layer_num: int,
    eh_sizes: list[int],
):
    origin_dict = {}
    delta_dict = {}

    for eh in eh_sizes:
        key = f"{eh}:{model_layer_num - eh}"

        in_name = f"input{input_id}.et_size{eh}.cb_input"
        out_name = f"input{input_id}.et_size{eh}.cb_output"

        tin = convert_to_numpy(tensors[in_name])
        tout = convert_to_numpy(tensors[out_name])

        origin_dict[key] = float(np.abs(tout).max())
        delta_dict[key] = float(np.abs(tout - tin).max())

    return origin_dict, delta_dict


# =========================
# 画图（只保留柱状图）
# =========================
def plot_dual_bar(
    origin_dict: dict,
    delta_dict: dict,
    save_path: str,
):
    labels = [k.split(":")[0] for k in origin_dict.keys()]
    x = np.arange(len(labels))
    width = 0.35
    split_ratio = 1  # 下方 4/5

    origin_vals = np.array(list(origin_dict.values()))
    delta_vals = np.array(list(delta_dict.values()))

    fig, ax = plt.subplots(figsize=(4, 3.2), layout="constrained")

    # ======================
    # Origin（左）
    # ======================
    origin_bottom = origin_vals * split_ratio  # 下方 4/5
    origin_top = origin_vals * (1 - split_ratio)  # 上方 1/5

    # 下方：斜线
    ax.bar(
        x - width / 2,
        origin_bottom,
        width,
        color=COLORS["Origin"],
        edgecolor="black",
        linewidth=1.0,
        hatch="///",
        zorder=2,
    )

    # 上方：纯色
    ax.bar(
        x - width / 2,
        origin_top,
        width,
        bottom=origin_bottom,
        color=COLORS["Origin"],
        edgecolor="black",
        linewidth=1.0,
        label="Origin",
        zorder=2,
    )

    # ======================
    # Delta（右）
    # ======================
    delta_bottom = delta_vals * split_ratio
    delta_top = delta_vals * (1 - split_ratio)

    # 下方：斜线
    ax.bar(
        x + width / 2,
        delta_bottom,
        width,
        color=COLORS["Delta"],
        edgecolor="black",
        linewidth=1.0,
        hatch="\\\\\\",
        zorder=2,
    )

    # 上方：纯色
    ax.bar(
        x + width / 2,
        delta_top,
        width,
        bottom=delta_bottom,
        color=COLORS["Delta"],
        edgecolor="black",
        linewidth=1.0,
        label="Delta",
        zorder=2,
    )

    # ======================
    # 坐标轴
    # ======================
    ax.set_ylabel("Data Range of Absolute Values", fontsize=14.3)
    ax.yaxis.set_label_coords(-0.19, 0.42)
    ax.set_xlabel("Number of Head and Tail Layers", fontsize=14.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks([200, 400, 600, 800, 1000, 1200, 1400, 1600])
    ax.set_ylim(0, 1650)

    ax.yaxis.grid(
        True, linestyle=":", linewidth=1.2, alpha=0.6, color="#BFBFBF", zorder=0
    )
    ax.xaxis.grid(False)

    origin_patch = mpatches.Patch(
    facecolor=COLORS["Origin"],
    edgecolor="black",
    hatch="///",
    label="Origin",
)

    delta_patch = mpatches.Patch(
        facecolor=COLORS["Delta"],
        edgecolor="black",
        hatch="\\\\\\",
        label="Residual",
    )

    ax.legend(
        handles=[origin_patch, delta_patch],
        loc="upper right",
        frameon=True,
    )

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =========================
# main
# =========================
def main():
    set_academic_style()

    input_id = 9
    model_layer_num = 20
    eh_sizes = [1, 3, 5, 7, 9, 11]

    tensors = load_file("/home/spl/workspace/zmx/dev/zllm/output/tensor_data/Wikitext/Qwen-3_wiki_prefill_tensors.safetensors")

    origin_dict, delta_dict = compute_origin_delta(
        tensors,
        input_id,
        model_layer_num,
        eh_sizes,
    )

    plot_dual_bar(
        origin_dict,
        delta_dict,
        save_path="4-3a.pdf",
    )

    print("✅ Saved to 4-3a.pdf")


if __name__ == "__main__":
    main()
