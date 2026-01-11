"""
Qwen-MoE LLM模拟器 - 使用重构后的模块
注意：此模型需要多卡，使用BaseEngineMulti
"""
from module.engine import BaseEngineMulti
from module.loader import WikitextLoader, TriviaQALoader
from run_test import run_test_suite


def main():
    # 配置
    TEST_MODEL = "qwen_moe"
    MODEL_PATH = "/home/share/models/Qwen3-30B-A3B" 
    WIKI_PATH = "datasets/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    TRIVIAQA_PATH = "datasets/trivia_qa-rc/validation*.parquet"
    TARGET_LAYERS = [4, 42]
    
    # 测试方法列表
    # 测试方法列表
    TARGET_TESTS = [
        # "Base", 
        # "INT4_AffDelta_Premix", 
        # "INT4_AffDelta_Mix_Group", 
        "Base",
        "INT2_Delta_Mix_Group", 
        "INT4_Delta_Mix_Group",
        "INT8_Delta_Mix_Group",
        # "INT4_Mix_Group",
        # "INT4_AffDelta_Group", 
        # "INT4_Delta_Group"
    ]
    
    # 初始化
    print("Initializing engine...")
    engine = BaseEngineMulti(MODEL_PATH, model_type=TEST_MODEL)
    engine.target_layer1 = TARGET_LAYERS[0]
    engine.target_layer2 = TARGET_LAYERS[1]
    
    # 初始化数据加载器
    print("Initializing data loader...")
    # loader = WikitextLoader(
    #     engine.tokenizer, 
    #     file_path=WIKI_PATH, 
    #     seq_len=1024, 
    #     model=TEST_MODEL
    # )
    loader = TriviaQALoader(
        engine.tokenizer,
        file_path=TRIVIAQA_PATH,
        use_context=True,
        model="qwen_moe"
    )

    test_types = ['f1']
    
    print("Running test suite...")
    results = run_test_suite(
        engine=engine,
        loader=loader,
        target_tests=TARGET_TESTS,
        target_layers=TARGET_LAYERS,
        model_name=TEST_MODEL,
        test_types=test_types,
        dataset="trvia_qa"
    )
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
