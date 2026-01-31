import json
import string
import re
import os
import glob
import fnmatch
from collections import Counter
import numpy as np
import pandas as pd

# ================= 1. 定义计算函数 (保持不变) =================
def clean_prediction_answer_prefix(s):
    """
    Remove Answer:/Answers:/answer: prefix from prediction string
    Checks after stripping leading whitespace
    """
    # Strip leading whitespace first
    s = s.lstrip()
    # Define patterns to match (case-insensitive)
    patterns = [r'^Answer:\s*', r'^Answers:\s*', r'^answer:\s*']
    # Apply each pattern
    for pattern in patterns:
        s = re.sub(pattern, '', s, flags=re.IGNORECASE)
    return s

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def compute_exact_match(prediction, truth):
    return int(normalize_answer(prediction) == normalize_answer(truth))

# ================= 2. 文件处理逻辑 =================

def format_filename(filename):
    """ 去掉后缀，并截掉最后两段（日期和时间） """
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.rsplit('_', 2)
    return parts[0]

def evaluate_single_file(file_path):
    f1_scores = []
    em_scores = []
    valid_samples = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if 'prediction' not in data or 'ground_truths' not in data:
                        continue
                    prediction = data['prediction']
                    # Clean prediction by removing Answer: prefixes
                    prediction = clean_prediction_answer_prefix(prediction)
                    ground_truths = data['ground_truths']
                    if isinstance(ground_truths, str):
                        ground_truths = [ground_truths]

                    f1 = metric_max_over_ground_truths(compute_f1, prediction, ground_truths)
                    em = metric_max_over_ground_truths(compute_exact_match, prediction, ground_truths)
                    
                    f1_scores.append(f1)
                    em_scores.append(em)
                    valid_samples += 1
                except json.JSONDecodeError:
                    continue
        
        if valid_samples == 0:
            return None

        raw_name = os.path.basename(file_path)
        clean_name = format_filename(raw_name)

        return {
            "Test": clean_name,
            "Samples": valid_samples,
            "Avg F1": round(np.mean(f1_scores) * 100, 2),
            "Exact Match": round(np.mean(em_scores) * 100, 2)
        }

    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return None

# ================= 3. 批量处理逻辑 (主要修改了这里) =================

def print_markdown_table(df):
    """
    输出 Markdown 格式表格
    """
    headers = ["Test", "Samples", "Avg F1", "Exact Match"]
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(['---'] * len(headers))} |")
    for _, row in df.iterrows():
        print(
            f"| {row['Test']} | "
            f"{int(row['Samples'])} | "
            f"{row['Avg F1']:.2f} | "
            f"{row['Exact Match']:.2f} |"
        )

def evaluate_files_with_pattern(file_pattern):
    """
    Evaluate files matching a glob pattern
    Args:
        file_pattern: Glob pattern (e.g., "output/f1/trvia_qa/glm*.json")
    """
    results = []
    print(f"Scanning files with pattern: {file_pattern} ...")
    
    # Use directory listing order when pattern targets a single directory
    matched_files = []
    base_dir = os.path.dirname(file_pattern) or "."
    pattern_name = os.path.basename(file_pattern)

    if "**" not in file_pattern and os.path.isdir(base_dir):
        try:
            for name in os.listdir(base_dir):
                if fnmatch.fnmatch(name, pattern_name):
                    matched_files.append(os.path.join(base_dir, name))
        except OSError:
            matched_files = glob.glob(file_pattern)
    else:
        # Use glob for recursive or complex patterns
        matched_files = glob.glob(file_pattern, recursive=True)

    # Sort by filename string order
    matched_files = sorted(matched_files, key=lambda p: os.path.basename(p))
    
    if not matched_files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(matched_files)} file(s)")
    
    for file_path in matched_files:
        stats = evaluate_single_file(file_path)
        if stats:
            results.append(stats)
    
    if results:
        df = pd.DataFrame(results)
        # 保持读取顺序
        df = df.reset_index(drop=True)

        # 输出 Markdown 表格
        print("")
        print_markdown_table(df)
        print("")
    else:
        print("No valid results computed.")

if __name__ == "__main__":
    # Example: Use glob pattern to match specific files
    # Pattern examples:
    # - "output/f1/trvia_qa/glm*.json" - files starting with glm
    # - "output/f1/narrative_qa/*.jsonl" - all jsonl files
    # - "output/f1/**/qwen*.json" - qwen files in any subdirectory
    
    pattern = "output/f1/trvia_qa/phi_INT8_P*.json"
    evaluate_files_with_pattern(pattern)
