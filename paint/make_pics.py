import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as path_effects
import json
import numpy as np

def plot_custom_grouped_bar(data_dict: dict, title="Percentage Comparison Chart", x_label="Data Labels", y_label="Percentage (%)", model="unknown"):
    # 1. 提取并排序横轴键
    sorted_x_keys = list(data_dict.keys())
    first_key = sorted_x_keys[0]
    group_names = list(data_dict[first_key].keys())
    
    if len(group_names) != 2:
        print("错误：第二层字典必须包含且仅包含两个分类名。")
        return

    # 2. 准备数据并计算均值
    y_vals = {name: [] for name in group_names}
    for x_key in sorted_x_keys:
        for name in group_names:
            vals = data_dict[x_key][name]
            avg = (sum(vals) / len(vals)) * 100 if vals else 0
            y_vals[name].append(avg)

    # 3. 样式配置 (更优雅的配色)
    color_palette = ['#7C9D97', '#EAB080'] # 深灰蓝 & 柔和金
    plt.style.use('seaborn-v0_8-whitegrid') # 使用简洁网格风格
    
    x_indexes = np.arange(len(sorted_x_keys))
    width = 0.38  # 稍微调整宽度
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    # 4. 绘制柱状图
    rects1 = ax.bar(x_indexes - width/2, y_vals[group_names[0]], width, 
                    label=group_names[0], color=color_palette[0], 
                    edgecolor='white', linewidth=1, zorder=3)
    rects2 = ax.bar(x_indexes + width/2, y_vals[group_names[1]], width, 
                    label=group_names[1], color=color_palette[1], 
                    edgecolor='white', linewidth=1, zorder=3)

    # 5. 坐标轴细化
    ax.set_ylim(20, 100) # 留出顶部空间给标注
    ax.set_ylabel(y_label, fontsize=11, color='#333333')
    ax.set_xlabel(x_label, fontsize=11, color='#333333')
    ax.set_title(title, fontsize=14, pad=20, fontweight='bold', color='#2C3E50')
    
    ax.set_xticks(x_indexes)
    ax.set_xticklabels([str(k) for k in sorted_x_keys])
    
    # 纵轴百分比格式
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # 隐藏上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # 6. 改进的标注函数 (防止重叠)
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            # 使用 path_effects 增加文字清晰度，防止重叠感
            txt = ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), # 向上偏移更多一点
                        textcoords="offset points",
                        ha='center', va='bottom', 
                        fontsize=9, fontweight='500', color='#2C3E50')
            txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    add_labels(rects1)
    add_labels(rects2)

    # 7. 图例放置在上方以节省空间
    ax.legend(frameon=True, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              ncol=2, fontsize=10)

    plt.tight_layout()
    plt.savefig(f'./output/similarity/{model}_{title.split(" ")[0].lower()}_similarity.png', bbox_inches='tight')

if __name__ == "__main__":
    TEST_MODEL = "qwen_moe"
    with open(f'./output/similarity/{TEST_MODEL}_prefill_similairty_analyze.json', 'r') as f:
        prefill_similarity_rcd = json.load(f)
    with open(f'./output/similarity/{TEST_MODEL}_decode_similairty_analyze.json', 'r') as f:
        decode_similarity_rcd = json.load(f)
        
    plot_custom_grouped_bar(prefill_similarity_rcd, title="Prefill Similarity", x_label="EH/ET Size", model = TEST_MODEL)
    plot_custom_grouped_bar(decode_similarity_rcd, title="Decode Similarity", x_label="Output Layer", model = TEST_MODEL)