"""
Qwen-MoE LLM模拟器 - 使用重构后的模块
注意：此模型需要多卡，使用BaseEngineMulti
"""
from module.engine import BaseEngineMulti
from module.loader import NarrativeQALoader, WikitextLoader, TriviaQALoader
from run_test import run_test_suite


def main():
    # 配置
    TEST_MODEL = "qwen_moe"
    TEST_DATASET = "wikitext"
    MODEL_PATH = "/home/share/models/Qwen3-30B-A3B" 
    WIKI_PATH = "datasets/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    TRIVIAQA_PATH = "datasets/trivia_qa-rc/validation*.parquet"
    NARRATIVEQA_PATH = "datasets/narrative_qa/test*.parquet"
    TARGET_LAYERS = [4, 42]

    TARGET_TESTS = [
        # "Base",
        # "INT2_Pertok",
        # "INT2_AffDelta_Group",
        "INT2_AffDelta_Mix_Group", 
        "INT2_Mix_Group",
        # "INT4_Pertok",
        # "INT4_AffDelta_Group",
        "INT4_AffDelta_Mix_Group",
        "INT4_Mix_Group",
        # "INT8_Pertok",
        # "INT8_AffDelta_Group",
        "INT8_AffDelta_Mix_Group",
        "INT8_Mix_Group",
    ]
    
    # 初始化
    print("Initializing engine...")
    engine = BaseEngineMulti(MODEL_PATH, model_type=TEST_MODEL)
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
    # loader = TriviaQALoader(
    #     engine.tokenizer,
    #     file_path=TRIVIAQA_PATH,
    #     use_context=True,
    #     model="qwen_moe"
    # )
    # loader = NarrativeQALoader(
    #     engine.tokenizer,
    #     file_path=NARRATIVEQA_PATH,
    #     use_context=True,
    #     model=TEST_MODEL
    # )

    test_types = ['ppl', 'size']
    
    print("Running test suite...")
    results = run_test_suite(
        engine=engine,
        loader=loader,
        target_tests=TARGET_TESTS,
        target_layers=TARGET_LAYERS,
        model_name=TEST_MODEL,
        test_types=test_types,
        dataset=TEST_DATASET
    )
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
