import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import pandas as pd
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] 
plt.rcParams['axes.unicode_minus'] = False


class AnalyzeTools:
    @staticmethod
    def calculate_tensor_similarity_with_position(tensor_a, tensor_b):
        """
        计算引入位置信息的相似度指标。
        输入应为相同形状的 NumPy 数组。
        """
        # 展平为一维向量，因为位置对应关系在展平后依然保持 (i 对应 i)
        flat_a = tensor_a.astype(np.float32).flatten()
        flat_b = tensor_b.astype(np.float32).flatten()

        # 1. 余弦相似度 (Cosine Similarity)
        # scipy 的 cosine 计算的是距离 (1 - similarity)，所以我们要用 1 减去它
        cos_sim = 1 - cosine(flat_a, flat_b)

        # 2. 皮尔逊相关系数 (Pearson Correlation)
        pearson_corr, _ = pearsonr(flat_a, flat_b)

        # 3. 欧氏距离 (L2 Norm of Difference) 和 相对距离
        l2_dist = np.linalg.norm(flat_a - flat_b)
        # 相对距离：距离除以 A 的模长，便于跨层比较
        relative_dist = l2_dist / (np.linalg.norm(flat_a) + 1e-9)

        return {
            "Cosine Similarity": cos_sim,
            "Pearson Correlation": pearson_corr,
            "Euclidean Distance": l2_dist,
            "Relative Distance": relative_dist
        }
    
    @staticmethod
    def analyze_outlier_overlap(tensor_a, tensor_b, sigma_threshold=3.0):
        """
        分析两个张量离群值的重合度和符号一致性，并计算重合离群值的平均差值。
        
        Args:
            tensor_a, tensor_b: 输入张量 (numpy array)
            sigma_threshold: 定义离群值的标准差倍数 (通常为 3.0)

        Returns:
            dict: 包含重合率、符号一致率、平均差值等指标的字典
        """
        # 1. 展平数据
        flat_a = tensor_a.astype(np.float32).flatten()
        flat_b = tensor_b.astype(np.float32).flatten()
        
        # 2. 计算统计量
        mean_a, std_a = np.mean(flat_a), np.std(flat_a)
        mean_b, std_b = np.mean(flat_b), np.std(flat_b)
        
        # 3. 离散化 (Discretization)
        disc_a = np.zeros_like(flat_a, dtype=np.int8)
        disc_b = np.zeros_like(flat_b, dtype=np.int8)
        
        # 标记 A 的离群值
        disc_a[flat_a > mean_a + sigma_threshold * std_a] = 1
        disc_a[flat_a < mean_a - sigma_threshold * std_a] = -1
        
        # 标记 B 的离群值
        disc_b[flat_b > mean_b + sigma_threshold * std_b] = 1
        disc_b[flat_b < mean_b - sigma_threshold * std_b] = -1
        
        # 4. 计算指标
        
        # 找出"至少有一个是离群值"的位置 (Union)
        union_mask = (disc_a != 0) | (disc_b != 0)
        # 找出"两个都是离群值"的位置 (Intersection)
        intersect_mask = (disc_a != 0) & (disc_b != 0)
        
        num_union = np.sum(union_mask)
        num_intersect = np.sum(intersect_mask)
        
        if num_intersect == 0:
            return {"Note": "没有发现重合的离群值位置"}

        # --- 指标 A: 离群值位置重合度 (Jaccard) ---
        jaccard_overlap = num_intersect / num_union if num_union > 0 else 0
        
        # --- 指标 B: 符号一致性 ---
        overlap_disc_a = disc_a[intersect_mask]
        overlap_disc_b = disc_b[intersect_mask]
        
        same_sign_count = np.sum(overlap_disc_a == overlap_disc_b)
        sign_consistency_rate = same_sign_count / num_intersect
        
        # --- [新增] 指标 C: 重合离群值的平均差值 ---
        # 提取重合位置的【原始数值】
        raw_val_a_overlap = flat_a[intersect_mask]
        raw_val_b_overlap = flat_b[intersect_mask]
        
        # 计算差值 (A - B)
        diffs = raw_val_a_overlap - raw_val_b_overlap
        
        # 计算平均差值 (Mean Difference)
        # 正值代表 A 在离群点通常比 B 大，负值代表 B 通常比 A 大
        avg_diff = np.mean(diffs)
        
        # (可选) 如果你需要的是差值的"幅度"平均（忽略正负抵消），可以用下面这行：
        # avg_abs_diff = np.mean(np.abs(diffs)) 
        
        return {
            "Outlier_Threshold": f"{sigma_threshold} sigma",
            "Total_Outliers_A": np.sum(disc_a != 0),
            "Total_Outliers_B": np.sum(disc_b != 0),
            "Overlapping_Outliers": num_intersect,
            "Position_Overlap (Jaccard)": jaccard_overlap, 
            "Sign_Consistency": sign_consistency_rate,
            # 新增的返回项
            "Average_Outlier_Diff": avg_diff
        }
    
    @staticmethod
    def analyze_outlier_overlap_percent(tensor_a, tensor_b, top_percent=1.0):
        """
        基于百分比（Top K% 绝对值）分析离群值重合度和符号一致性。

        Args:
            tensor_a, tensor_b: 输入张量 (numpy array)
            top_percent: 定义离群值的百分比，例如 1.0 代表绝对值最大的前 1%

        Returns:
            dict: 包含阈值、重合率、符号一致率等指标的字典
        """
        # 1. 展平并转为 float32 以防溢出
        flat_a = tensor_a.astype(np.float32).flatten()
        flat_b = tensor_b.astype(np.float32).flatten()
        
        # 2. 计算基于绝对值的百分位阈值
        # 例如 top 1%，即寻找第 99 百分位数
        q = 100 - top_percent
        
        threshold_a = np.percentile(np.abs(flat_a), q)
        threshold_b = np.percentile(np.abs(flat_b), q)
        
        # 3. 离散化 (Discretization)
        # 0: 正常, 1: 正离群, -1: 负离群
        disc_a = np.zeros_like(flat_a, dtype=np.int8)
        disc_b = np.zeros_like(flat_b, dtype=np.int8)
        
        # 标记 A (大于阈值为正离群，小于负阈值为负离群)
        disc_a[flat_a > threshold_a] = 1
        disc_a[flat_a < -threshold_a] = -1
        
        # 标记 B
        disc_b[flat_b > threshold_b] = 1
        disc_b[flat_b < -threshold_b] = -1
        
        # 4. 计算指标
        
        # Union: 只要有一方是离群值
        union_mask = (disc_a != 0) | (disc_b != 0)
        # Intersection: 两方同时是离群值 (位置重合)
        intersect_mask = (disc_a != 0) & (disc_b != 0)
        
        num_union = np.sum(union_mask)
        num_intersect = np.sum(intersect_mask)
        
        if num_intersect == 0:
            return {
                "Info": f"在 Top {top_percent}% 中未发现位置重合的离群值",
                "Threshold_A": threshold_a,
                "Threshold_B": threshold_b
            }

        # --- 指标 A: 位置重合度 (Jaccard Index) ---
        jaccard_overlap = num_intersect / num_union if num_union > 0 else 0
        
        # --- 指标 B: 符号一致性 (Sign Consistency) ---
        # 提取重合位置的符号
        overlap_signs_a = disc_a[intersect_mask]
        overlap_signs_b = disc_b[intersect_mask]
        
        # 检查符号是否相同 (1对1, 或 -1对-1)
        same_sign_count = np.sum(overlap_signs_a == overlap_signs_b)
        sign_consistency_rate = same_sign_count / num_intersect
        
        return {
            "Definition": f"Top {top_percent}% (Abs Value)",
            "Threshold_Value_A": threshold_a,
            "Threshold_Value_B": threshold_b,
            "Total_Outliers_Expected": int(flat_a.size * (top_percent / 100)),
            "Overlapping_Count": num_intersect,
            # 核心指标
            "Position_Overlap (Jaccard)": jaccard_overlap,
            "Sign_Consistency": sign_consistency_rate
        }

    @staticmethod
    def display_dict_as_adaptive_table(data: Dict[Any, Dict[Any, Any]], title: Optional[str] = "Data Table") -> None:
        df = pd.DataFrame(data).T 

        df = df.fillna('')
        
        print(f"--- {title} ---")
        print(df.to_markdown())

    @staticmethod
    def plot_custom_grouped_bar(data_dict, title="Percentage Comparison Chart", x_label="Data Labels", y_label="Percentage (%)"):
        """
        根据特定字典结构绘制分组柱状图。
        
        :param data_dict: 格式为 {int: {str: list[float]}}
                        例如: {101: {"A组": [0.1, 0.2], "B组": [0.3, 0.4]}}
        """
        # 1. 提取并排序横轴键 (int)
        sorted_x_keys = sorted(data_dict.keys())
        
        # 2. 提取分类名称 (str) 
        # 取第一个数据项的第二层字典键作为分类名
        first_key = sorted_x_keys[0]
        group_names = list(data_dict[first_key].keys())
        
        if len(group_names) != 2:
            print("错误：第二层字典必须包含且仅包含两个分类名。")
            return

        # 3. 准备数据
        y_vals_group1 = []
        y_vals_group2 = []
        
        for x_key in sorted_x_keys:
            # 计算第一组的均值
            list1 = data_dict[x_key][group_names[0]]
            avg1 = (sum(list1) / len(list1)) * 100 if list1 else 0
            y_vals_group1.append(avg1)
            
            # 计算第二组的均值
            list2 = data_dict[x_key][group_names[1]]
            avg2 = (sum(list2) / len(list2)) * 100 if list2 else 0
            y_vals_group2.append(avg2)

        # 4. 设置绘图参数
        x_indexes = np.arange(len(sorted_x_keys)) # 横轴基准位置
        width = 0.35 # 柱子宽度
        
        plt.figure(figsize=(10, 6))
        plt.clf()

        # 5. 绘制两组柱子
        # 使用自定义颜色，并从字典中提取组名作为 label
        rects1 = plt.bar(x_indexes - width/2, y_vals_group1, width, 
                        label=group_names[0], color='#4E79A7', edgecolor='white')
        rects2 = plt.bar(x_indexes + width/2, y_vals_group2, width, 
                        label=group_names[1], color='#F28E2B', edgecolor='white')

        # 6. 坐标轴与格式设置
        plt.ylim(0, 100) # 固定纵轴 0-100%
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        
        # 设置横坐标刻度
        plt.xticks(x_indexes, [str(k) for k in sorted_x_keys])
        
        # 纵轴显示百分比符号
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        # 7. 添加图例和上方数值标注
        plt.legend()

        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                plt.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        add_labels(rects1)
        add_labels(rects2)

        plt.tight_layout()
        plt.savefig(f'{title}.png')
        print(f"图表已生成并保存为 '{title}.png'")

    @staticmethod
    def calculate_tensor_entropy(tensor):
        # 1. 确保数据是 numpy 格式并扁平化 (模拟论文中的分组统计)
        if hasattr(tensor, 'detach'):
            data = tensor.detach().cpu().numpy().flatten()
        else:
            data = np.array(tensor).flatten()

        # 2. 如果是连续值，通常需要先进行量化 (Quantization)
        # 论文中是针对量化后的离散符号进行统计的。
        # 这里假设输入已经是量化后的离散值，或者通过简单的取整模拟量化。
        # 如果已经是离散符号，可跳过这一步。
        
        # 3. 统计各符号出现的频率
        # 使用 np.unique 统计每个唯一值出现的次数
        _, counts = np.unique(data, return_counts=True)
        
        # 4. 计算概率分布
        probabilities = counts / len(data)
        
        # 5. 应用香农熵公式 (使用以 2 为底的对数，单位为 bits)
        # 加上微小的 offset (1e-12) 避免 log(0) 错误
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        
        return entropy
