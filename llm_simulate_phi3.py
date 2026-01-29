"""
GLM-4 LLM模拟器 - 使用重构后的模块
"""
from module.engine import BaseEngineMulti
from module.loader import WikitextLoader, TriviaQALoader, NarrativeQALoader
from run_test import run_test_suite


def main():
    # 配置
    TEST_MODEL = "phi"
    TEST_DATASET = "wikitext"
    MODEL_PATH = "/home/share/models/Phi-3.5-MoE-instruct"
    WIKI_PATH = "datasets/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    TRIVIAQA_PATH = "datasets/trivia_qa-rc/validation*.parquet"
    NARRATIVEQA_PATH = "datasets/narrative_qa/test*.parquet"
    TARGET_LAYERS = [4, 26]
    
    # 初始化
    print("Initializing engine...")
    engine = BaseEngineMulti(MODEL_PATH, model_type = TEST_MODEL)
    engine.target_layer1 = TARGET_LAYERS[0]
    engine.target_layer2 = TARGET_LAYERS[1]
    
    ##########################narrative_qa测试##########################
    loader = NarrativeQALoader(
        engine.tokenizer,
        file_path=NARRATIVEQA_PATH,
        use_context=True,
        model=TEST_MODEL
    )
    TEST_DATASET = "narrative_qa"
    
    TARGET_TESTS = [
        "INT4_MixGp_Group",
        "INT4_AffDelta_Group",
        "INT8_Pertok",
        "INT4_Pertok",
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
        "INT4_AffDelta_Group",
        "INT8_Pertok",
        "INT4_Pertok",
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
