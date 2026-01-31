import argparse
import csv
import glob
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


def read_similarity_csv(csv_path):
    layer_pairs = []
    similarities = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_pairs.append(row["layer_pair"])
            similarities.append(float(row["cosine_similarity"]))
    return layer_pairs, similarities


def extract_dataset_name(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    return parts[0] if len(parts) > 1 else base


def collect_model_data(model_dir):
    csv_files = sorted(glob.glob(os.path.join(model_dir, "*.csv")))
    datasets = OrderedDict()
    x_labels = None

    for csv_path in csv_files:
        dataset_name = extract_dataset_name(csv_path)
        layer_pairs, similarities = read_similarity_csv(csv_path)
        if x_labels is None:
            x_labels = layer_pairs
        datasets[dataset_name] = (layer_pairs, similarities)

    return x_labels, datasets


def plot_multi_model_similarity(root_dir, save_path, figsize=(8, 3)):
    model_dirs = sorted(
        d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)
    )
    if not model_dirs:
        raise FileNotFoundError(f"No model folders found in {root_dir}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.unicode_minus": False,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
    })

    fig, axes = plt.subplots(
        1,
        len(model_dirs),
        figsize=figsize,
        layout="constrained",
        sharey=True,
    )

    if len(model_dirs) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []
    legend_y = -0.01
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0, 0.18, 1, 0.8))

    threshold_value = 0.7
    threshold_color = "#D62728"

    line_colors = ["#EE9817", "#1F77B4", "#2CA02C"]

    for idx, (ax, model_dir) in enumerate(zip(axes, model_dirs)):
        model_name = os.path.basename(model_dir)
        x_labels, datasets = collect_model_data(model_dir)
        if not datasets:
            continue

        if x_labels is None:
            continue

        x_positions = np.arange(len(x_labels))

        for data_idx, (dataset_name, (layer_pairs, similarities)) in enumerate(datasets.items()):
            if layer_pairs != x_labels:
                lookup = {lp: sim for lp, sim in zip(layer_pairs, similarities)}
                aligned = [lookup.get(lp, np.nan) for lp in x_labels]
            else:
                aligned = similarities

            (line,) = ax.plot(
                x_positions,
                aligned,
                color=line_colors[data_idx % len(line_colors)],
                linewidth=1.6,
                marker="o",
                markersize=3.5,
                label=dataset_name,
            )

            if dataset_name not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(dataset_name)

        ax.set_title(model_name, fontsize=12, fontweight="bold")
        tick_indices = np.arange(0, len(x_labels), 4)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(
            [x_labels[i] for i in tick_indices],
            rotation=35,
            ha="center",
            # rotation_mode="anchor",
        )

        ax.axhline(
            y=threshold_value,
            color=threshold_color,
            linestyle=(0, (6, 6)),
            linewidth=1.0,
            zorder=1,
        )
        if idx == 3:
            text_x = 0.98
            text_ha = "right"
            text_y = threshold_value - 0.025
            text_va = "top"
            text_transform = ax.get_yaxis_transform()
            ax.text(
            text_x,
            text_y,
            "strong similarity",
            transform=text_transform,
            ha=text_ha,
            va=text_va,
            fontsize=9,
            color=threshold_color,
            fontweight="bold",
        )
        else:
            # text_x = 0.98
            # text_ha = "right"
            # text_y = threshold_value - 0.025
            # text_va = "top"
            # text_transform = ax.get_yaxis_transform()
            pass
        
        ax.grid(axis="y", linestyle=(0, (4, 6)), linewidth=0.6, color="#B0B0B0")

        for spine in ax.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("Cosine Similarity", fontsize=14)
    fig.supxlabel(
        r"Split Point $\langle k_{1}, k_{2} \rangle$",
        fontsize=14,
        y=0.11,
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=max(1, len(legend_labels)),
        frameon=False,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot multi-model cosine similarity line charts"
    )
    parser.add_argument(
        "--root_dir",
        default="./output/tensor_data",
        help="Root directory containing model folders",
    )
    parser.add_argument(
        "--save_path",
        default="./model_cosine_similarity.pdf",
        help="Output figure path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_multi_model_similarity(args.root_dir, args.save_path)
