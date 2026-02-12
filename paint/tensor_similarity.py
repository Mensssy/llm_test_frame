import csv
import glob
import os

import numpy as np
import torch
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from safetensors.torch import save_file, load_file

def plot_similarity_chart(data_dict, title, save_path=None, figsize=(4, 3.5), dpi=300):
    # 1. 全局样式高级化定制
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Lato', 'Arial', 'DejaVu Sans'], # 优先选择更现代的字体
        'axes.unicode_minus': False,
        'axes.facecolor': '#FFFFFF',
        'figure.facecolor': '#FFFFFF',
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=100, layout='constrained')
    
    # 数据准备
    x_labels = list(data_dict.keys())
    y_values = list(data_dict.values())
    x_positions = np.arange(len(x_labels))
    
    # 2. 配色方案：深石墨色 + 柔和金/青蓝 (此处采用低饱和度的深青色)
    main_color = "#21568A"     # 优雅的深石墨蓝
    fill_color = '#3498DB'     # 填充色
    threshold_color = "#D84123" # 柔和的珊瑚红
    grid_color = '#F0F0F0'
    
    # 3. 绘制带有阴影的面积折线图 (增加呼吸感)
    ax.fill_between(x_positions, y_values, color=fill_color, alpha=0.08, zorder=2)
    
    # 绘制主折线
    line, = ax.plot(x_positions, y_values, 
                    color=main_color, 
                    linewidth=1.6,
                    marker='.',
                    markersize=9,
                    markerfacecolor='white',
                    markeredgecolor=main_color,
                    markeredgewidth=2,
                    label='Similarity Score',
                    zorder=4)
    
    # 4. 优化参考线：更细、更淡
    # threshold = 0.7
    # ax.axhline(y=threshold, color=threshold_color, linestyle=(0, (5, 5)), linewidth=1.0, alpha=1, zorder=1)
    
    # # 参考线文本：移动到左侧或右侧边缘，避免干扰
    # ax.text(len(x_positions)-0.5, threshold - 0.07, f'Strong Similarity ({threshold})', 
    #         color=threshold_color, fontsize=11, ha='right', fontweight='540', zorder=100)

    # 5. 坐标轴与细节微调
    ax.set_ylim(0.0, 1.1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='left', fontsize=13, color='#1A1A1A')
    # 获取所有的标签
    labels = [item.get_text() for item in ax.get_xticklabels()]

    # 每隔一个显示，其余变为空字符串
    new_labels = [label if i % 2 == 0 else "" for i, label in enumerate(labels)]

    ax.set_xticklabels(new_labels)
    
    # 移除顶部和右侧边框，左侧和底部改用极浅色
    # for spine in ['top', 'right']:
    #     ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_color("#000000")
        ax.spines[spine].set_linewidth(1.0)

    # 仅保留水平网格线
    ax.yaxis.grid(True, linestyle='-', color=grid_color, zorder=0)
    ax.xaxis.grid(False)

    # 标题与标签
    # ax.set_title(title, pad=25, fontsize=16, fontweight='bold', color="#000000", loc='left')
    ax.tick_params(axis='y', labelsize=12, width=1, color='#1A1A1A')
    ax.set_ylabel('Cosine Similarity', labelpad=7, fontsize=16, color="#000000")
    ax.set_xlabel(r'Split Point $\langle k_1, k_2 \rangle$', labelpad=7, fontsize=16, color="#000000")
    
    # 增加内边距
    
    # plt.tight_layout()

    # 6. 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    
    return fig

def plot_dual_bar_chart(data_dict, title, save_path=None, figsize=(4, 3.5), dpi=300):
    # 1. 全局样式高级化
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=100, layout='constrained')
    
    # 数据提取
    tags = list(data_dict.keys())
    delta_values = [data_dict[tag]["Delta"] for tag in tags]
    original_values = [data_dict[tag]["Origin"] for tag in tags]
    
    x = np.arange(len(tags))
    width = 0.38  # 稍微加宽柱子，显得更厚重踏实
    
    # 2. 配色方案：深邃蓝 (Original) 与 优雅灰 (Delta)
    # 这种对比强调了“原始数据”为主，“变化量”为辅的逻辑
    color_main = "#275182"
    color_sub = "#AD3B24"
    
    bars1 = ax.bar(x - width/2, original_values, width, 
                   label='Origin', color=color_main, 
                   zorder=3, alpha=1)
    
    bars2 = ax.bar(x + width/2, delta_values, width, 
                   label='Residual', color=color_sub, 
                   zorder=3, alpha=1)

    # 4. Y 轴优化 (加大数字)
    ax.tick_params(axis='y', labelsize=13, colors='#000000', length=0)
    ax.tick_params(axis='x', labelsize=11, colors='#000000', length=4)
    
    # 动态计算最大值并设置 y 轴范围
    all_vals = original_values + delta_values
    max_val = max(all_vals) if all_vals else 1.0
    ax.set_ylim(0, max_val * 1.35) 

    # 5. 标题与标签
    # 标题左对齐，更有设计感
    # ax.set_title(title, pad=30, fontsize=18, fontweight='bold', color='#1A1A1A', loc='left')
    ax.set_ylabel('Data Range', labelpad=7, fontsize=16, color="#000000")

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=0, ha='center') # 如果标签不长，建议不旋转
    ax.set_xlabel(r'Split Point $<k1, k2>$', labelpad=7, fontsize=16, color="#000000")
    # 6. 精简边框与网格
    ax.spines['top'].set_color("#000000")
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    
    # 仅开启水平网格线，且颜色极浅
    ax.yaxis.grid(True, linestyle='-', color='#F2F2F2', zorder=0)

    # 7. 图例优化：移至图表下方
    ax.legend(
        loc='upper center',          # 图例相对于锚点的位置
        bbox_to_anchor=(0.24, 0.98), # 锚点坐标 (x, y)，(0.5, 0) 是底部正中心，-0.15 是向下偏移量
        ncol=1,                      # 设置为 2 列，使其水平排列
        frameon=True,               # 移除边框，保持简洁
        fontsize=12,
        handletextpad=0.5,           # 图标与文字的距离
        columnspacing=0.7            # 两组图例之间的间距
    )

    # 注意：因为图例移到了外部下方，可能需要调整底部的留白
    # plt.tight_layout(rect=[0, 0.05, 1, 1])

    # plt.tight_layout()

    # 8. 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    
    plt.close(fig)
    return fig

def convert_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    if tensor.requires_grad:
        tensor = tensor.detach() 

    return tensor.numpy()

def calculate_tensor_similarity(tensor_a, tensor_b):
    # similarity, r = calculate_outlier_overlap(tensor_a, tensor_b)
    # print(r)
    # return similarity
    # 展平为一维向量，因为位置对应关系在展平后依然保持 (i 对应 i)
    flat_a = tensor_a.astype(np.float32).flatten()
    flat_b = tensor_b.astype(np.float32).flatten()

    # 1. 余弦相似度 (Cosine Similarity)
    # scipy 的 cosine 计算的是距离 (1 - similarity)，所以我们要用 1 减去它
    cos_sim = 1 - cosine(flat_a, flat_b)

    # 2. 皮尔逊相关系数 (Pearson Correlation)
    pearson_corr, _ = pearsonr(flat_a, flat_b)

    # return {
    #     "Cosine Similarity": cos_sim,
    #     "Pearson Correlation": pearson_corr,
    # }
    return cos_sim

def calculate_outlier_overlap(tensor_a, tensor_b, ratio=0.0005):
    """
    计算两个张量离群值（绝对值最大的前 ratio 比例的数据）的位置重合率，
    以及重合部分正负号相同的概率。
    """
    # 展平为一维向量
    flat_a = tensor_a.astype(np.float32).flatten()
    flat_b = tensor_b.astype(np.float32).flatten()
    
    n = len(flat_a)
    if n == 0:
        return 0.0, 0.0
        
    k = int(n * ratio)
    if k == 0:
        k = 1  # 至少取 1 个点，防止除零或无法计算
    
    # argpartition 可以高效找到最大的 k 个元素位置（无序）
    # indices of top k elements by absolute value
    top_k_idx_a = np.argpartition(np.abs(flat_a), -k)[-k:]
    top_k_idx_b = np.argpartition(np.abs(flat_b), -k)[-k:]
    
    # 计算位置重合的索引
    intersection = np.intersect1d(top_k_idx_a, top_k_idx_b)
    overlap_count = len(intersection)
    
    # 1. 重合率 (Overlap Rate)
    overlap_rate = overlap_count / k
    
    # 2. 正负号一致性概率 (Sign Consistency Probability)
    sign_consistency = 0.0
    if overlap_count > 0:
        # 获取重合位置上的原始符号
        signs_a = np.sign(flat_a[intersection])
        signs_b = np.sign(flat_b[intersection])
        
        # 统计符号相同的数量
        match_count = np.sum(signs_a == signs_b)
        sign_consistency = match_count / overlap_count
        
    return overlap_rate, sign_consistency


def compute_input_delta_similarity_rows(
    tensors,
    input_id,
    model_layer_num,
    eh_sizes,
):
    rows = []
    for eh_size in eh_sizes:
        input_tensor_name = f"input{input_id}.et_size{eh_size}.cb_input"
        output_tensor_name = f"input{input_id}.et_size{eh_size}.cb_output"

        input_tensor = tensors[input_tensor_name]
        output_tensor = tensors[output_tensor_name]

        similarity = calculate_tensor_similarity(
            convert_to_numpy(input_tensor),
            convert_to_numpy(output_tensor),
        )

        rows.append(
            {
                "eh_size": eh_size,
                "layer_pair": f"{eh_size}:{model_layer_num - eh_size}",
                "input_tensor_name": input_tensor_name,
                "output_tensor_name": output_tensor_name,
                "cosine_similarity": float(similarity),
            }
        )
    return rows


def write_similarity_csv(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "eh_size",
        "layer_pair",
        "input_tensor_name",
        "output_tensor_name",
        "cosine_similarity",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
if __name__ == "__main__":
    # input_id = 4
    # model_layer_num = 64

    # eh_sizes = list(range(1, 15, 1))
    # input_dir = "./output/tensor_data/LongChat"
    # input_files = sorted(
    #     glob.glob(os.path.join(input_dir, "phi*.safetensors"))
    # )
    # # if not input_files:
    # #     input_files = [
    # #         "./output/tensor_data/qwen_prefill_tensors.safetensors",
    # #     ]

    # for safetensor_path in input_files:
    #     tensors = load_file(safetensor_path)
    #     rows = compute_input_delta_similarity_rows(
    #         tensors=tensors,
    #         input_id=input_id,
    #         model_layer_num=model_layer_num,
    #         eh_sizes=eh_sizes,
    #     )

    #     base_name = os.path.splitext(os.path.basename(safetensor_path))[0]
    #     csv_path = os.path.join(
    #         input_dir,
    #         f"{base_name}_similarity.csv",
    #     )
    #     write_similarity_csv(rows, csv_path)
    print("请确保在A800-2上运行")
    
    input_id = 9
    model_layer_num = 48
    tensors = load_file("/home/spl/workspace/zmx/dev/zllm/output/tensor_data/Wikitext/Qwen-3_wiki_prefill_tensors.safetensors")
    
    data_dict = {}
    for eh_size in range(1,13,1):
        input_tensor_name = f"input{input_id}.et_size{eh_size}.cb_input"
        output_tensor_name = f"input{input_id}.et_size{eh_size}.cb_output"
        data_dict[f"{eh_size - 1 + 1}:{model_layer_num - eh_size}"] = calculate_tensor_similarity(convert_to_numpy(tensors[input_tensor_name]), convert_to_numpy(tensors[output_tensor_name]))
    plot_similarity_chart(data_dict, "Cosine Similarity between Different 2 Layer", "./4-3b.pdf")
    
    data_dict = {}
    for eh_size in range(1,13,2):
        input_tensor_name = f"input{input_id}.et_size{eh_size}.cb_input"
        output_tensor_name = f"input{input_id}.et_size{eh_size}.cb_output"
        data_dict[f"{eh_size - 1 + 1}:{model_layer_num - eh_size}"] = {}
        data_dict[f"{eh_size - 1 + 1}:{model_layer_num - eh_size}"] = {
            "Origin": tensors[output_tensor_name].abs().max().item(),
            "Delta": (tensors[output_tensor_name] - tensors[input_tensor_name]).abs().max().item(),
        }
    plot_dual_bar_chart(data_dict, "Max Abs Value Comparison", "./4-3a.pdf")