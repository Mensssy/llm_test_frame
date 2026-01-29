from safetensors.torch import save_file, load_file
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

def plot_continuous_ccdf(tensor, use_log_x=True):
    """
    绘制无截断的连续 CCDF 分布图
    
    Args:
        tensor: 输入张量
        use_log_x: 是否对X轴也使用对数刻度？
                   - True: 推荐。Transformer激活通常跨度极大(0~10000+)，双对数坐标(Log-Log)能呈现完美的线性幂律分布。
                   - False: 线性X轴。如果离群值非常远，主体部分会被压缩在左侧边缘。
    """
    # 1. 数据预处理
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().float().numpy().flatten()
    else:
        data = np.array(tensor).flatten()
    
    data = np.abs(data)
    total_count = len(data)
    
    # 2. 设置风格
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": "--"})
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 3. 绘制 CCDF (Complementary CDF)
    # complementary=True 意味着绘制 P(X >= x)
    # 这里的关键是：一张图，画到底
    sns.ecdfplot(data=data, ax=ax, color='#4C72B0', linewidth=2.5, complementary=True)
    
    # 4. 设置坐标轴
    ax.set_yscale('log') # Y轴保持 Log Scale
    ax.set_ylabel("CCDF: $P(X \geq x)$ (Log Scale)", fontsize=12)
    
    if use_log_x:
        ax.set_xscale('log')
        ax.set_xlabel("Activation Abs Values (Log Scale)", fontsize=12, fontweight='bold')
        # 处理 Log X 轴可能遇到的 0 值问题（自动调整下限）
        non_zero_min = data[data > 0].min() if len(data[data > 0]) > 0 else 1e-3
        ax.set_xlim(left=non_zero_min)
    else:
        ax.set_xlabel("Activation Abs Values (Log Scale)", fontsize=12, fontweight='bold')
        ax.set_xlim(left=0)

    # 5. 关键点标注 (Max & Percentiles)
    # 标注最大值
    max_val = data.max()
    min_prob = 1.0 / total_count
    
    ax.scatter([max_val], [min_prob], color='#C44E52', s=50, zorder=5, edgecolor='white', label='Max Value')
    ax.annotate(f'Max: {max_val:.1f}\n$P \\approx 10^{{{int(np.log10(min_prob))}}}$', 
                 xy=(max_val, min_prob), 
                 xytext=(-30, 15), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1),
                 ha='right', fontsize=10, fontweight='bold', color='#C44E52')

    # 标注 99% 分位数 (展示主体数据的边界)
    p99 = np.percentile(data, 99)
    p99_prob = 0.01 # 因为是CCDF，99%分位对应纵轴 0.01 (1-0.99)
    
    # 在曲线上标记 p99
    ax.scatter([p99], [p99_prob], color='#55A868', s=40, zorder=5, edgecolor='white')
    ax.text(p99, p99_prob * 1.5, f' 99% < {p99:.1f}', color='#55A868', fontsize=10, fontweight='bold')
    
    plt.xlim(left = 0.1)

    # 6. 标题与排版
    scale_info = "Log-Log Scale" if use_log_x else "Log-Linear Scale"
    plt.title(f"Activation Distribution CCDF({scale_info})\nTotal Elements: {total_count:,}", 
              fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig("tensor_ccdf.png", dpi=300)
    print("Plot saved as tensor_ccdf.png")
    
def plot_smooth_ccdf(tensor, use_log_x=True, smoothing_sigma=2):
    """
    绘制极度平滑的 CCDF 分布图
    通过在 Log 空间重采样和高斯滤波消除毛刺
    """
    # 1. 数据准备
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().float().numpy().flatten()
    else:
        data = np.array(tensor).flatten()
    data = np.abs(data)
    total_count = len(data)
    
    # 排序并计算原始 CDF
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y_raw = np.arange(n, 0, -1) / n
    
    # --- 核心优化：对数空间重采样 ---
    # 目的：不管原始数据多密集或多稀疏，强制生成固定数量(比如500个)均匀分布的点
    
    # 1. 确定范围（避免log(0)）
    min_val = data[data > 0].min() if len(data[data > 0]) > 0 else 1e-4
    max_val = data.max()
    
    # 2. 生成在对数轴上均匀分布的 X 坐标 (500个点足够平滑了)
    x_smooth = np.geomspace(min_val, max_val, num=500)
    
    # 3. 插值：将原始的不均匀数据映射到均匀的 X 坐标上
    # fill_value="extrapolate" 处理边界情况
    f = interpolate.interp1d(sorted_data, y_raw, kind='linear', bounds_error=False, fill_value=(1.0, 0.0))
    y_smooth = f(x_smooth)
    
    # 4. (可选) 高斯平滑：进一步消除微小的抖动
    if smoothing_sigma > 0:
        y_smooth = gaussian_filter1d(y_smooth, sigma=smoothing_sigma)
        
    # --- 绘图 ---
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": "--"})
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # 绘制平滑曲线
    ax.plot(x_smooth, y_smooth, color='#4C72B0', lw=3, label='Smoothed Trend')

    # 保留最大值的真实点（因为平滑可能会把极端的最大值“磨”掉，需要单独标出来）
    # real_max_val = sorted_data[-1]
    # real_min_prob = y_raw[-1]
    # ax.scatter([real_max_val], [real_min_prob], color='#C44E52', s=50, zorder=5, edgecolor='white')
    
    # 坐标轴设置
    ax.set_yscale('log')
    ax.set_ylabel("CCDF: $P(X \geq x)$ (Log Scale)", fontsize=12)
    
    if use_log_x:
        ax.set_xscale('log')
        ax.set_xlabel("Activation Abs Values (Log Scale)", fontsize=12, fontweight='bold')
        ax.set_xlim(left=min_val * 0.9)
    else:
        ax.set_xlabel("Activation Abs Values (Log Scale)", fontsize=12, fontweight='bold')
        ax.set_xlim(left=0)
        
    plt.xlim(left = 0.1)
    # 标注
    # ax.annotate(f'Max: {real_max_val:.1f}', 
    #              xy=(real_max_val, real_min_prob), 
    #              xytext=(-40, -20), textcoords='offset points',
    #              arrowprops=dict(arrowstyle='->', color='black', lw=1),
    #              ha='right', fontsize=10, fontweight='bold', color='#C44E52')

    scale_info = "Log-Log Scale" if use_log_x else "Log-Linear Scale"
    plt.title(f"Activation Smooth Distribution CCDF({scale_info})\nTotal Elements: {total_count:,}", 
              fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig("tensor_smooth_ccdf.png", dpi=300)
    print("Plot saved as tensor_smooth_ccdf.png")
    
def _process_single_tensor(tensor, smoothing_sigma=2):
    """
    内部辅助函数：处理单个 Tensor，返回平滑后的曲线数据和极值信息
    """
    # 1. 基础预处理
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().float().numpy().flatten()
    else:
        data = np.array(tensor).flatten()
    data = np.abs(data)
    
    # 2. 排序与原始概率计算
    sorted_data = np.sort(data)
    n = len(sorted_data)
    y_raw = np.arange(n, 0, -1) / n
    
    # 3. 对数空间重采样 (Log-Space Resampling)
    # 确定有效范围 (防止 log(0))
    min_val = data[data > 0].min() if len(data[data > 0]) > 0 else 1e-4
    max_val = data.max()
    
    # 生成均匀的对数坐标 X
    x_smooth = np.geomspace(min_val, max_val, num=500)
    
    # 插值
    f = interpolate.interp1d(sorted_data, y_raw, kind='linear', bounds_error=False, fill_value=(1.0, 0.0))
    y_smooth = f(x_smooth)
    
    # 4. 高斯平滑
    if smoothing_sigma > 0:
        y_smooth = gaussian_filter1d(y_smooth, sigma=smoothing_sigma)
        
    # 返回：平滑后的X, 平滑后的Y, 真实最大值, 真实最大值对应的概率
    return x_smooth, y_smooth, sorted_data[-1], y_raw[-1]

def plot_dual_smooth_ccdf(tensor1, tensor2, 
                          label1="Tensor A", label2="Tensor B", 
                          use_log_x=True, smoothing_sigma=2):
    """
    对比绘制两个 Tensor 的超平滑 CCDF 分布图
    """
    # 1. 分别处理两个 Tensor
    # 这里完全复刻了之前的逻辑，只是执行了两次
    x1, y1, max1, prob1 = _process_single_tensor(tensor1, smoothing_sigma)
    x2, y2, max2, prob2 = _process_single_tensor(tensor2, smoothing_sigma)
    
    # 2. 设置绘图风格
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": "--"})
    plt.figure(figsize=(12, 7)) # 稍微加宽一点，方便放图例
    ax = plt.gca()

    # 定义颜色方案 (Seaborn Deep Palette)
    color1 = '#4C72B0' # 蓝色
    color2 = '#DD8452' # 橙色 (对比度高)

    # 3. 绘制曲线
    ax.plot(x1, y1, color=color1, lw=3, label=f'{label1}')
    ax.plot(x2, y2, color=color2, lw=3, label=f'{label2}')

    # 4. 绘制并标注最大值 (真实数据点)
    # Tensor 1 Max
    ax.scatter([max1], [prob1], color=color1, s=60, zorder=5, edgecolors='white', marker='o')
    ax.annotate(f'Max: {max1:.1f}', xy=(max1, prob1), 
                xytext=(-10, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=color1, lw=1.5),
                ha='right', fontsize=9, fontweight='bold', color=color1)

    # Tensor 2 Max
    ax.scatter([max2], [prob2], color=color2, s=60, zorder=5, edgecolors='white', marker='D') # 菱形标记区分
    ax.annotate(f'Max: {max2:.1f}', xy=(max2, prob2), 
                xytext=(10, 20),           # 坐标改为正数：(向右偏移, 向上偏移)
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=color2, lw=1.5),
                ha='left',                 # 文本左对齐，这样文字会显示在点的右侧
                fontsize=9, fontweight='bold', color=color2)

    # 5. 坐标轴设置
    ax.set_yscale('log')
    ax.set_ylabel("CCDF: $P(X \geq x)$", fontsize=12)
    
    # 确定 X 轴范围 (取两个 tensor 中最小的 min 和 最大的 max)
    global_min = min(x1[0], x2[0])
    
    if use_log_x:
        ax.set_xscale('log')
        ax.set_xlabel("Activation Abs Values (Log Scale)", fontsize=12, fontweight='bold')
        ax.set_xlim(left=global_min * 0.8) # 留点余地
    else:
        ax.set_xlabel("Activation Abs Values (Linear Scale)", fontsize=12, fontweight='bold')
        ax.set_xlim(left=0)
    
    plt.xlim(left = 1)

    # 6. 图例与标题
    # 添加半透明背景的图例
    plt.legend(frameon=True, fontsize=11, loc='upper right', framealpha=0.9)
    
    plt.title(f"Activation Distribution Comparison\n{label1} vs {label2}", 
              fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig("tensor_dual_ccdf.png", dpi=300)
    print("Plot saved as tensor_dual_ccdf.png")

# --- 使用示例 ---
if __name__ == "__main__":
    tensors = load_file("output/tensor_data/qwen_prefill_tensors.safetensors")
    # 画一个Tensor，曲线不作平滑
    # use_log_x 代表x轴是否使用log scale
    plot_continuous_ccdf(tensors["input3.et_size7.cb_output"], use_log_x=True)
    
    # 画一个Tensor
    # plot_smooth_ccdf(tensors["input3.et_size7.cb_output"], use_log_x=True)
    
    # 画两个Tensor
    # plot_dual_smooth_ccdf(tensors["input3.et_size7.cb_output"], tensors["input3.et_size7.cb_output"] - tensors["input3.et_size7.cb_input"],
    #                       label1="CB Output", label2="CB Delta",
    #                       use_log_x=True)