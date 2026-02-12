import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def plot_benchmark_charts(data_list, title_list=None, y_label_list=None, first_log_scale=False, second_k_format=False):
    """
    根据传入的3个字典、3个标题和3个Y轴标签绘制柱状图。
    
    参数:
    data_list (list): 包含3个字典的列表。每个字典的 Key 是 X轴标签，Value 是 Y轴数值。
    title_list (list): 包含3个字符串的列表，对应每个图表的标题。
    y_label_list (list): 包含3个字符串的列表，对应每个图表的Y轴标签。
    first_log_scale (bool): 第一个子图是否使用对数坐标 (Log Scale)
    """
    
    # 检查数据完整性
    if len(data_list) != 3 or len(y_label_list) != 3:
        raise ValueError("请确保传入了3个字典数据、3个标题和3个Y轴标签。")

    # 定义复刻图片中的配色 (蓝, 绿, 红)
    colors = ['#4e79a7', '#59a14f', '#e15759']
    
    # 创建画布 (1行3列)
    # [调整] 使用 layout="constrained" 自动管理布局，解决标签长度不同导致的视觉间距不一致问题
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), dpi=100, layout="constrained")
    
    # 调整子图之间的间距 (增加wspace以防不同长度的Y轴标签重叠)
    # [移除] 使用 constrained layout 后不再需要手动调整 subplots_adjust
    # plt.subplots_adjust(wspace=0.4, bottom=0.22, top=0.98, left=0.09, right=0.99)
    
    # 如果觉得自动布局的间距太紧，可以通过 set_constrained_layout_pads 设置 padding
    # w_pad 控制子图间的水平最小间距，h_pad 控制垂直间距 (在单行图中不明显)
    fig.get_layout_engine().set(w_pad=0.1, h_pad=0, wspace=0.05)

    # 遍历每个子图进行绘制
    # 使用 zip 同时遍历 axes, data_list, title_list, y_label_list, colors
    for i, (ax, data, title, y_label, color) in enumerate(zip(axes, data_list, title_list, y_label_list, colors)):
        keys = list(data.keys())
        values = list(data.values())
        x_pos = np.arange(len(keys))
        
        # 1. 绘制柱状图
        # [调整] width 控制柱子的粗细。该值越小，柱子越细，柱间空隙越大。这里从 0.65 改为 0.5
        bars = ax.bar(x_pos, values, color=color, edgecolor='black', linewidth=1.5, width=0.7)
        
        # 2. 设置标题
        # ax.set_title(title, fontsize=16, pad=15, linespacing=1.4)
        
        # [修改点] 设置对应的 Y轴标签
        ax.set_ylabel(y_label, fontsize=16)
        
        # [新增] 设置 Y 轴刻度字体大小
        ax.tick_params(axis='y', labelsize=13)
        
        # [新增] 针对第二张图设置 k 单位格式
        if i == 1 and second_k_format:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*1e-3:g}k' if x != 0 else '0'))
        
        # 3. 设置X轴标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(keys, rotation=28, ha='right', rotation_mode="anchor", fontsize=16)
        
        # 4. 在柱子上方添加数值文本
        y_max = max(values) if values else 1
        
        # 针对第一张图应用 Log Scale
        if i == 0 and first_log_scale:
            ax.set_yscale('log')
            # 保持柱子上方数据标签的一致性，在 log scale 下通常需要调整 ylim 上限以防止标签被截断
            # ax.set_ylim(bottom=min(values)*0.5, top=y_max * 10) 
            # 或者仅设置 top，bottom 会自动
            ax.set_ylim(top=y_max * 7)
            # [新增] 移除对数坐标轴的细小刻度 (Minor Ticks)，只保留 10^n 对应的主刻度
            ax.minorticks_off() 
        else:
            # [调整] 增加 Y 轴上限的倍数，从 1.25 改为 1.35，给顶部数值留出更多空间
            ax.set_ylim(0, y_max * 1.35) 
        
        for bar in bars:
            height = bar.get_height()
            
            # 使用科学计数法显示大数值
            if height >= 10000: 
                label_text = f"{height:.1e}".replace("e+0", "e").replace("e+", "e")
            else:
                label_text = f"{height}"

            # [调整] 调整字体大小，防止重叠
            # Log scale 下文本位置可能需要微调，这里保持逻辑一致，偏移量可能需要根据视觉效果微调
            # 如果是 log scale，y_max * 0.02 可能太小了，但在 log 坐标下 linear 的 offset 表现不同
            # 简单起见，这里依然沿用之前的逻辑，但在 log scale 下，文本位置是由数据坐标决定的
            offset = height * 0.05 if (i==0 and first_log_scale) else y_max * 0.02
            
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                    label_text,
                    ha='center', va='bottom', fontsize=12, fontweight="normal")
            
        # 5. 添加背景网格
        # [调整] 显式指定 only major ticks 的网格，避免 log scale 下出现细密的网格线
        ax.yaxis.grid(True, which='major', linestyle='--', alpha=0.5, color='lightgrey')
        ax.set_axisbelow(True)

        # 加粗坐标轴边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    plt.savefig("TTFT_TPOT_Throughput.pdf")

# 1. 数据
dict1 = {"Plain Text": 116.9, "CAP": 170.3, "SCX": 3670.8, "CPU-TEE": 2320261.7}
dict2 = {"Plain Text": 23.2, "CAP": 100.8, "SCX": 4805.9, "CPU-TEE": 187.9}
dict3 = {"Plain Text": 193.9, "CAP": 193, "SCX": 115.0, "CPU-TEE": 19.7}

data_list = [dict1, dict2, dict3]

# 2. 标题
titles = [
    "", 
    "", 
    ""
]

# 3. [修改点] 不同的 Y轴标签列表
y_labels = [
    "TTFT (ms)",
    "TPOT (ms)",
    "Throughput (tokens/s)",
]

# 4. 调用函数
# 设置 first_log_scale=True 来启用第一张图的对数坐标
# 设置 second_k_format=True 来启用第二张图的 k 单位格式
plot_benchmark_charts(data_list, titles, y_labels, first_log_scale=True, second_k_format=True)