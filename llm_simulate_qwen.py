"""
Qwen LLM模拟器 - 使用重构后的模块
"""
from module.engine import BaseEngineMulti
from module.loader import WikitextLoader
from run_test import run_test_suite


def main():
    # 配置
    TEST_MODEL = "qwen"
    MODEL_PATH = "/home/share/models/Qwen2.5-32B-Instruct" 
    WIKI_PATH = "datasets/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    TARGET_LAYERS = [7, 55]
    
    # 测试方法列表
    TARGET_TESTS = [
        "Base", 
        # "INT4_AffDelta_Premix", 
        # "INT4_AffDelta_Mix_Group", 
        # "INT4_Delta_Mix_Group", 
        # "INT4_AffDelta_Group", 
        # "INT4_Delta_Group",
        # "INT4_Mix_Group",
    ]
    
    # 初始化
    print("Initializing engine...")
    engine = BaseEngineMulti(MODEL_PATH)
    engine.target_layer1 = TARGET_LAYERS[0]
    engine.target_layer2 = TARGET_LAYERS[1]
    
    # 初始化数据加载器
    print("Initializing data loader...")
    loader = WikitextLoader(
        engine.tokenizer, 
        file_path=WIKI_PATH, 
        seq_len=1024, 
        model=TEST_MODEL
    )
    
    # 运行测试套件
    # 可选的测试类型: 'f1', 'ppl', 'size', 'entropy', 'similarity', 'tensor_save'
    test_types = ['tensor_save']
    
    print("Running test suite...")
    results = run_test_suite(
        engine=engine,
        loader=loader,
        target_tests=TARGET_TESTS,
        target_layers=TARGET_LAYERS,
        model_name=TEST_MODEL,
        test_types=test_types
    )
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
