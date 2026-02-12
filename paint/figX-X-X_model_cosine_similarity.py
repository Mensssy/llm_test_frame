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
            layer_pairs.append(row["eh_size"])
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


def plot_multi_model_similarity(root_dir, save_path, figsize=(4, 2.8)):
    all_model_dirs = {
        os.path.basename(d): d
        for d in glob.glob(os.path.join(root_dir, "*"))
        if os.path.isdir(d)
    }
    desired_dataset_order = ["LongChat", "TriviaQA", "NarrativeQA", "Wikitext"]
    model_dirs = []
    for name in desired_dataset_order:
        if name in all_model_dirs:
            model_dirs.append(all_model_dirs[name])
    
    # If explicit list is empty (e.g. folder names don't match), fall back to sorted list
    if not model_dirs:
         model_dirs = sorted(
            d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)
        )

    if not model_dirs:
        raise FileNotFoundError(f"No model folders found in {root_dir}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "axes.unicode_minus": False,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })

    threshold_value = 0.7
    threshold_color = "#D62728"

    line_colors = ["#EE9817", "#1F77B4", "#2CA02C"]

    label_map = {
        "Qwen-3": "Qwen3-30B-A3B",
        "Phi-3.5": "Phi-3.5-MoE",
        "GLM-4": "GLM-4-9B",
    }
    
    # Desired order for legend
    desired_order = ["Qwen3-30B-A3B", "Phi-3.5-MoE", "GLM-4-9B"]

    for idx, model_dir in enumerate(model_dirs):
        model_name = os.path.basename(model_dir)
        x_labels, datasets = collect_model_data(model_dir)
        if not datasets:
            continue

        if x_labels is None:
            continue
            
        # Create a separate figure for each model
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

        x_positions = np.arange(len(x_labels))
        
        legend_handles = []
        legend_labels = []

        for data_idx, (dataset_name, (layer_pairs, similarities)) in enumerate(datasets.items()):
            display_name = label_map.get(dataset_name, dataset_name)
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
                label=display_name,
            )

            if display_name not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(display_name)

        # ax.set_title(model_name, fontsize=12, fontweight="normal")
        ax.set_xlabel("Number of Head and Tail Layers", fontsize=15)
        ax.set_ylabel("Cosine Similarity", fontsize=16)
        
        tick_indices = np.arange(0, len(x_labels), 3)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(
            [x_labels[i] for i in tick_indices],
            rotation=0,
            ha="center",
        )
        
        ax.grid(axis="y", linestyle=(0, (4, 6)), linewidth=0.6, color="#B0B0B0")

        for spine in ax.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(1.0)
            
        # Legend Sorting and Placement
        sorted_handles = []
        sorted_labels = []

        handle_map = dict(zip(legend_labels, legend_handles))

        for label in desired_order:
            if label in handle_map:
                sorted_labels.append(label)
                sorted_handles.append(handle_map[label])

        # Add any others
        for label in legend_labels:
            if label not in sorted_labels:
                sorted_labels.append(label)
                sorted_handles.append(handle_map[label])

        ax.legend(
            sorted_handles,
            sorted_labels,
            loc="lower right",
            frameon=True
        )

        # Construct specific save path
        dir_name = os.path.dirname(save_path)
        base_name = os.path.basename(save_path)
        name_root, ext = os.path.splitext(base_name)
        new_filename = f"{name_root}_{model_name}{ext}"
        final_path = os.path.join(dir_name, new_filename)
        
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        fig.savefig(final_path, dpi=300)
        plt.close(fig)
        print(f"Saved figure: {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot multi-model cosine similarity line charts"
    )
    parser.add_argument(
        "--root_dir",
        default="/home/spl/workspace/zmx/dev/zllm/output/tensor_data",
        help="Root directory containing model folders",
    )
    parser.add_argument(
        "--save_path",
        default="./model_cosine_similarity.pdf",
        help="Output figure path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("请确保在A800-2上运行")
    args = parse_args()
    plot_multi_model_similarity(args.root_dir, args.save_path)
