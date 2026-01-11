import numpy as np
import torch
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from safetensors.torch import save_file, load_file

def plot_similarity_chart(data_dict, title, save_path=None, figsize=(5, 4), dpi=300):
    # 1. 全局样式高级化定制
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Lato', 'Arial', 'DejaVu Sans'], # 优先选择更现代的字体
        'axes.unicode_minus': False,
        'axes.facecolor': '#FFFFFF',
        'figure.facecolor': '#FFFFFF',
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
    # 数据准备
    x_labels = list(data_dict.keys())
    y_values = list(data_dict.values())
    x_positions = np.arange(len(x_labels))
    
    # 2. 配色方案：深石墨色 + 柔和金/青蓝 (此处采用低饱和度的深青色)
    main_color = "#21568A"     # 优雅的深石墨蓝
    fill_color = '#3498DB'     # 填充色
    threshold_color = '#E74C3C' # 柔和的珊瑚红
    grid_color = '#F0F0F0'
    
    # 3. 绘制带有阴影的面积折线图 (增加呼吸感)
    ax.fill_between(x_positions, y_values, color=fill_color, alpha=0.08, zorder=2)
    
    # 绘制主折线
    line, = ax.plot(x_positions, y_values, 
                    color=main_color, 
                    linewidth=2.5,
                    marker='.',
                    markersize=9,
                    markerfacecolor='white',
                    markeredgecolor=main_color,
                    markeredgewidth=2,
                    label='Similarity Score',
                    zorder=4)
    
    # 4. 优化参考线：更细、更淡
    threshold = 0.7
    ax.axhline(y=threshold, color=threshold_color, linestyle=(0, (5, 5)), linewidth=1.5, alpha=1, zorder=1)
    
    # 参考线文本：移动到左侧或右侧边缘，避免干扰
    ax.text(len(x_positions)-0.5, threshold + 0.02, f'Strong Similarity ({threshold})', 
            color=threshold_color, fontsize=11, ha='right', fontweight='700')

    # 5. 坐标轴与细节微调
    ax.set_ylim(0.0, 1.1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, ha='left', fontsize=12, color='#1A1A1A')
    # 获取所有的标签
    labels = [item.get_text() for item in ax.get_xticklabels()]

    # 每隔一个显示，其余变为空字符串
    new_labels = [label if i % 2 == 0 else "" for i, label in enumerate(labels)]

    ax.set_xticklabels(new_labels)
    
    # 移除顶部和右侧边框，左侧和底部改用极浅色
    # for spine in ['top', 'right']:
    #     ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom', 'top', 'right']:
        ax.spines[spine].set_color("#616161")
        ax.spines[spine].set_linewidth(0.8)

    # 仅保留水平网格线
    ax.yaxis.grid(True, linestyle='-', color=grid_color, zorder=0)
    ax.xaxis.grid(False)

    # 标题与标签
    # ax.set_title(title, pad=25, fontsize=16, fontweight='bold', color="#000000", loc='left')
    ax.tick_params(axis='y', labelsize=12, width=1, color='#1A1A1A')
    ax.set_ylabel('Cosine Similarity', labelpad=7, fontsize=17, color="#000000")
    ax.set_xlabel('A:B Layers', labelpad=7, fontsize=15, color="#000000")
    
    # 增加内边距
    plt.tight_layout()

    # 6. 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.show()
    
    return fig

def plot_dual_bar_chart(data_dict, title, save_path=None, figsize=(5, 4), dpi=300):
    # 1. 全局样式高级化
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    
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
                   label='Delta', color=color_sub, 
                   zorder=3, alpha=1)

    # 4. Y 轴优化 (加大数字)
    ax.tick_params(axis='y', labelsize=13, colors='#000000', length=0)
    ax.tick_params(axis='x', labelsize=11, colors='#000000', length=4)
    # 定义格式化函数：x 是数值，pos 是位置
    def format_k(x, pos):
        return f'{x * 1e-3:g}k'

    # 应用到 y 轴
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_k))
    
    # 动态计算最大值并设置 y 轴范围
    all_vals = original_values + delta_values
    max_val = max(all_vals) if all_vals else 1.0
    ax.set_ylim(0, max_val * 1.1) 

    # 5. 标题与标签
    # 标题左对齐，更有设计感
    # ax.set_title(title, pad=30, fontsize=18, fontweight='bold', color='#1A1A1A', loc='left')
    ax.set_ylabel('Max Abs Value', labelpad=7, fontsize=17, color="#000000")

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=0, ha='center') # 如果标签不长，建议不旋转
    ax.set_xlabel('A:B Layers', labelpad=7, fontsize=15, color="#000000")

    # 6. 精简边框与网格
    ax.spines['top'].set_color("#616161")
    ax.spines['right'].set_color('#616161')
    ax.spines['left'].set_color('#616161')
    ax.spines['bottom'].set_color('#616161')
    
    # 仅开启水平网格线，且颜色极浅
    ax.set_ylim(0, 21300)
    ax.yaxis.grid(True, linestyle='-', color='#F2F2F2', zorder=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))

    # 7. 图例优化：移至图表下方
    ax.legend(
        loc='upper center',          # 图例相对于锚点的位置
        bbox_to_anchor=(0.15, 0.98), # 锚点坐标 (x, y)，(0.5, 0) 是底部正中心，-0.15 是向下偏移量
        ncol=1,                      # 设置为 2 列，使其水平排列
        frameon=True,               # 移除边框，保持简洁
        fontsize=10,
        handletextpad=0.5,           # 图标与文字的距离
        columnspacing=0.7            # 两组图例之间的间距
    )

    # 注意：因为图例移到了外部下方，可能需要调整底部的留白
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.tight_layout()

    # 8. 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
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
    
if __name__ == "__main__":
    input_id = 7
    model_layer_num = 64
    tensors = load_file("./datasets/qwen_prefill_tensors.safetensors")
    
    data_dict = {}
    for eh_size in range(1,13,1):
        input_tensor_name = f"input{input_id}.et_size{eh_size}.cb_input"
        output_tensor_name = f"input{input_id}.et_size{eh_size}.cb_output"
        data_dict[f"{eh_size - 1 + 1}:{model_layer_num - eh_size}"] = calculate_tensor_similarity(convert_to_numpy(tensors[input_tensor_name]), convert_to_numpy(tensors[output_tensor_name]))
        
    plot_similarity_chart(data_dict, "Cosine Similarity between Different 2 Layer", "./output/similarity/layers_similarity.pdf")
    
    data_dict = {}
    for eh_size in range(1,13,2):
        input_tensor_name = f"input{input_id}.et_size{eh_size}.cb_input"
        output_tensor_name = f"input{input_id}.et_size{eh_size}.cb_output"
        data_dict[f"{eh_size - 1 + 1}:{model_layer_num - eh_size}"] = {}
        data_dict[f"{eh_size - 1 + 1}:{model_layer_num - eh_size}"] = {
            "Origin": tensors[output_tensor_name].abs().max().item(),
            "Delta": (tensors[output_tensor_name] - tensors[input_tensor_name]).abs().max().item(),
        }
    plot_dual_bar_chart(data_dict, "Max Abs Value Comparison", "./output/similarity/layers_max_abs_comparison.pdf")