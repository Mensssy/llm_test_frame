# LLM Precision Test Framework

## 简介
本项目是一个用于Transformer结构大模型精度测试的框架。通过对模型中任意层进行量化、剪枝或其他操作，模拟不同的压缩策略，并评估其对模型输出结果（精度）以及层间数据传输量的影响。

## 主要功能
- **多模型支持**: 支持 Llama, GLM-4, Qwen, Qwen-MoE, Phi-3 等主流开源大模型。
- **量化模拟**: 提供灵活的量化策略模拟，包括 INT8, INT4, INT2 等不同精度，以及 Group, Mix, Affine Delta 等多种量化方案。
- **多维评估**:
    - **PPL (Perplexity)**: 困惑度测试，评估语言模型能力。
    - **F1 Score**: 针对问答任务（如 TriviaQA, NarrativeQA）的精度评分。
    - **传输量测试**: 分析层间数据传输大小，评估压缩效果。
    - **相似度分析**: 计算量化前后张量的余弦相似度。
    - **离群值分析**: 支持 Global TopK 和 Group TopK 等离群值检测与分离策略。
- **可视化**: `paint/` 目录下包含丰富的绘图脚本，可用于生成相似度热图、吞吐量图表等。

## 项目结构
```
.
├── llm_simulate_*.py       # 各模型的模拟启动脚本 (Llama, GLM4, Qwen, Phi3, etc.)
├── run_test.py             # 统一测试运行主要逻辑 (F1, PPL, Size Test)
├── module/                 # 核心功能模块
│   ├── engine.py           # 推理引擎 (BaseEngineMulti)，支持多卡推理
│   ├── loader.py           # 数据集加载器 (Wikitext, TriviaQA, NarrativeQA, LongChat)
│   ├── edit_tools.py       # 量化与张量编辑工具 (Outlier extraction, Quantization)
│   └── analyze_tools.py    # 分析工具
├── evaluate/               # 评测脚本 (如 eval_f1.py)
├── paint/                  # 绘图与可视化脚本
├── datasets/               # 数据集存放目录 (需自行准备)
└── output/                 # 实验结果输出目录
```

## 使用说明

### 1. 配置模型与数据
在 `llm_simulate_<model>.py` 脚本中配置模型路径与数据集路径：

```python
MODEL_PATH = "/path/to/your/model"
WIKI_PATH = "datasets/wikitext-2-raw-v1/..."
```

### 2. 配置测试参数
在启动脚本中定义需要测试的层 (`TARGET_LAYERS`) 和测试方法 (`TARGET_TESTS`)：

```python
# 目标层
TARGET_LAYERS = [4, 42]

# 测试策略列表
TARGET_TESTS = [
    "Base",                 # 基准
    "INT8_Mix_Group",       # INT8 混合分组量化
    "INT4_AffDelta_Group",  # INT4 仿射Delta分组量化
    ...
]
```

### 3. 运行测试
运行对应的模型模拟脚本：

```bash
python llm_simulate_llama.py
# 或
python llm_simulate_glm4.py
```

### 4. 查看结果
测试结果将以 JSON 格式保存在 `output/` 目录下，可视化结果位于 `paint/` 输出或其他指定位置。

## 注意事项
- 本项目依赖本地模型文件，请确保 `MODEL_PATH` 指向正确的权重路径。
- 部分模型（如 Qwen-MoE）使用 `BaseEngineMulti`，会自动利用多张 GPU 进行推理，请确保显存充足。
- 数据集文件通常为 Parquet 格式，需预先下载至 `datasets/` 目录。
