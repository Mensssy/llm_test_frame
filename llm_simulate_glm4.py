"""
GLM-4 LLM模拟器 - 使用重构后的模块
"""
from module.engine import BaseEngineMulti
from module.loader import WikitextLoader, TriviaQALoader, NarrativeQALoader
from run_test import run_test_suite


def main():
    # 配置
    TEST_MODEL = "glm"
    TEST_DATASET = "narrative_qa"
    MODEL_PATH = "/home/share/models/glm-4-9b-chat-1m"
    MODEL_PATH = "/home/share/models/GLM-4-9B-0414"
    WIKI_PATH = "datasets/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    TRIVIAQA_PATH = "datasets/trivia_qa-rc/validation*.parquet"
    NARRATIVEQA_PATH = "datasets/narrative_qa/test*.parquet"
    TARGET_LAYERS = [4, 34]
    
    # 测试方法列表
    TARGET_TESTS = [
        # "Base",
        # "INT4_Pertok",
        # "INT4_AffDelta_Group",
        # "INT4_AffDelta_Mix_Group",
        # "INT4_Mix_Group",
        # "INT8_Pertok",
        # "INT8_AffDelta_Group",
        # "INT8_AffDelta_Mix_Group",
        # "INT8_Mix_Group",
        # "INT2_Pertok",
        # "INT2_AffDelta_Group",
        # "INT2_AffDelta_Mix_Group", 
        # "INT2_Mix_Group",
    ]
    
    # 初始化
    print("Initializing engine...")
    engine = BaseEngineMulti(MODEL_PATH, model_type = TEST_MODEL)
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
    # loader = TriviaQALoader(
    #     engine.tokenizer,
    #     file_path=TRIVIAQA_PATH,
    #     use_context=True,
    #     model=TEST_MODEL
    # )

    ##########################narrative_qa测试##########################
    loader = NarrativeQALoader(
        engine.tokenizer,
        file_path=NARRATIVEQA_PATH,
        use_context=True,
        model=TEST_MODEL
    )
    TEST_DATASET = "narrative_qa"
    
    # INT4_MixGp
    TARGET_TESTS = [
        "INT4_MixGp_Group",
    ]
    test_types = ['f1']
    engine.OUTLIER_METHOD = "group_topk"
    
    print("Running test suite...")
    results = run_test_suite(
        engine=engine,
        loader=loader,
        target_tests=TARGET_TESTS,
        target_layers=TARGET_LAYERS,
        model_name=TEST_MODEL,
        test_types=test_types,
        output_len=64,
        dataset=TEST_DATASET
    )
    
    # INT4_AffDelta_Mix
    TARGET_TESTS = [
        "INT4_AffDelta_Mix_Group",
    ]
    test_types = ['f1']
    engine.OUTLIER_METHOD = "global_topk"
    
    print("Running test suite...")
    results = run_test_suite(
        engine=engine,
        loader=loader,
        target_tests=TARGET_TESTS,
        target_layers=TARGET_LAYERS,
        model_name=TEST_MODEL,
        test_types=test_types,
        output_len=64,
        dataset=TEST_DATASET
    )
    
    ##########################trivia_qa测试##########################
    loader = TriviaQALoader(
        engine.tokenizer,
        file_path=TRIVIAQA_PATH,
        use_context=True,
        model=TEST_MODEL
    )
    TEST_DATASET = "trvia_qa"
    TARGET_TESTS = [
        "INT4_MixGp_Group",
    ]
    test_types = ['f1']
    engine.OUTLIER_METHOD = "group_topk"
    
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
    
    # INT4_AffDelta_Mix
    TARGET_TESTS = [
        "INT4_AffDelta_Mix_Group",
    ]
    test_types = ['f1']
    engine.OUTLIER_METHOD = "global_topk"
    
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
        
    
    #################################size测试#################################
    loader = WikitextLoader(
        engine.tokenizer, 
        file_path=WIKI_PATH, 
        seq_len=1024, 
        model=TEST_MODEL
    )
    TEST_DATASET = "wikitext"
    TARGET_TESTS = [
        "Base",
        "INT4_AffDelta_MixGp_Group",
        "INT8_Pertok",
        "INT4_Pertok",
    ]
    test_types = ['size']
    engine.OUTLIER_METHOD = "group_topk"
    for l in [2048, 8192]:
        loader.seq_len = l
        
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
        
    #################################save测试#################################
    loader = WikitextLoader(
        engine.tokenizer, 
        file_path=WIKI_PATH, 
        seq_len=1024, 
        model=TEST_MODEL
    )
    TEST_DATASET = "wikitext"
    TARGET_TESTS = [
        "Base",
    ]
    test_types = ['tensor_save']
    
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
