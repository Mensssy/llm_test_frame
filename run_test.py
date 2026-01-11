"""
统一的测试运行模块
包含F1测试、PPL测试、传输大小测试、熵测试、相似度测试和张量保存测试
"""
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from safetensors.torch import save_file
from module.analyze_tools import AnalyzeTools


class TestRunner:
    def __init__(self, engine, loader, model_name="model", dataset="wiki"):
        """
        Args:
            engine: BaseEngineSingle 或 BaseEngineMulti 实例
            loader: WikitextLoader 或 TriviaQALoader 实例
            model_name: 模型名称，用于输出文件命名
        """
        self.engine = engine
        self.loader = loader
        self.model_name = model_name
        self.dataset = dataset
        self.record = {}
    
    def run_f1_test(self, target_tests, target_layers, output_len=16, max_samples=None):
        """
        F1分数测试
        Args:
            target_tests: 测试方法列表
            target_layers: 目标层列表 [layer1, layer2]
            output_len: 生成长度
            max_samples: 最大测试样本数
        """
        print(f"\n>>> F1 Score Test...")
        
        for test in target_tests:
            print(f">>> Running {test} test...")
            edit_config = {
                target_layers[0]: "INT8_Mix_Group" if test != "Base" else "Base",
                target_layers[1]: test
            }
            self.engine.register_layer_hooks(edit_config)
            
            output_file = f"./output/f1/{self.dataset}/{self.model_name}_{test}_{datetime.now().strftime('%m-%d_%H%M')}.json"
            with open(output_file, "w", encoding='utf-8') as f:
                for i, sample in enumerate(self.loader):
                    if max_samples and i >= max_samples:
                        break
                        
                    prediction = self.engine.generate(sample["input_ids"], max_new_tokens=output_len)
                    
                    record = {
                        "id": sample["id"],
                        "prediction": prediction,
                        "ground_truths": sample['ground_truths'],
                    }
                    
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
            
            self.engine.remove_hooks()
            print(f">>> Finished {test} test.")
    
    def run_ppl_test(self, target_tests, target_layers, max_length=2048):
        """
        困惑度测试
        Args:
            target_tests: 测试方法列表
            target_layers: 目标层列表 [layer1, layer2]
            max_length: 最大上下文长度
        """
        print(f"\n>>> PPL Test...")
        
        for test in target_tests:
            print(f">>> Running {test} test...")
            edit_config = {
                target_layers[0]: "INT8_Mix_Group" if test != "Base" else "Base",
                target_layers[1]: test
            }
            self.engine.register_layer_hooks(edit_config)
            
            edited_ppl = self.engine.calculate_perplexity(self.loader.input_ids, max_length=max_length)
            print(f"{test} PPL: {edited_ppl:.4f}")
            self.record[test] = self.record.get(test, {})
            self.record[test]["PPL"] = edited_ppl
            
            self.engine.remove_hooks()
    
    def run_size_test(self, target_tests, target_layers):
        """
        传输大小测试
        Args:
            target_tests: 测试方法列表
            target_layers: 目标层列表 [layer1, layer2]
        """
        print(f"\n>>> Trans Size Test...")
        data_iter = self.loader.get_batches()
        input_ids = next(data_iter)
        
        for test in target_tests:
            print(f">>> Running {test} test...")
            edit_config = {
                target_layers[0]: "INT8_Mix_Group" if test != "Base" else "Base",
                target_layers[1]: test
            }
            self.engine.register_layer_hooks(edit_config)
            
            prediction = self.engine.generate(input_ids, 1)  # 只需要测prefill
            self.record[test] = self.record.get(test, {})
            self.record[test]["Size"] = self.engine.hid2_size
            
            self.engine.remove_hooks()
            print(f">>> Finished {test} test.")
    
    
    def run_tensor_save_test(self, input_ids_num=10, max_eh_et_len=20, 
                             decode_test_layers=None, max_decode_len=10,
                             save_decode=False):
        """
        张量保存测试
        Args:
            input_ids_num: 测试输入数量
            max_eh_et_len: Prefill阶段的最大EH/ET长度
            decode_test_layers: Decode阶段的测试层列表（当save_decode=True时必须提供）
            max_decode_len: 最大decode长度
            save_decode: 是否保存decode张量
        """
        if save_decode and decode_test_layers is None:
            raise ValueError("当save_decode=True时，必须提供decode_test_layers参数来指定要保存哪些layer的output")
        data_iter = self.loader.get_batches()
        
        prefill_tensor_rcd = {}
        decode_tensor_rcd = {}
        
        for i in range(input_ids_num):
            print(f">>>>>>>>>> Input {i} <<<<<<<<<<", flush=True)
            input_ids = next(data_iter)
        
            print("\n>>> Generate Prefill Data...")
            for l in range(1, max_eh_et_len+1):
                self.engine.target_layer1 = l - 1
                self.engine.target_layer2 = self.engine.layer_nums - l - 1
                edit_config = {
                    l - 1: "INT8_Mix_Group",
                    self.engine.layer_nums - l - 1: "Base",
                }
                self.engine.register_layer_hooks(edit_config)
                self.engine.generate(input_ids, 1)
                
                prefill_tensor_rcd[f"input{i}.et_size{l}.cb_input"] = self.engine.hid1
                prefill_tensor_rcd[f"input{i}.et_size{l}.cb_output"] = self.engine.hid2
                
                print(f">>> Finished {l} test.", flush=True)
            self.engine.remove_hooks()
            
            if save_decode:
                print("\n>>> Generate Decode Data...", flush=True)
                print(f"Saving decode outputs for layers: {decode_test_layers}", flush=True)
                for l in decode_test_layers:
                    self.engine.target_layer1 = 0
                    self.engine.target_layer2 = l
                    edit_config = {
                        0: "INT8_Mix_Group",
                        l: "Base",
                    }
                    self.engine.register_layer_hooks(edit_config)
                    self.engine.generate(input_ids, max_decode_len+1, enable_hid_rcd=True)

                    for idx in range(1, len(self.engine.hid_rcd)):
                        decode_tensor_rcd[f"input{i}.layer{l}.iter{idx-1}"] = self.engine.hid_rcd[idx]["layer2_hid"]
                        
                    self.engine.remove_hooks()
                    print(f">>> Finished {l} test.", flush=True)
                    
        save_file(prefill_tensor_rcd, f"output/tensor_data/{self.model_name}_prefill_tensors.safetensors")
        if save_decode:
            save_file(decode_tensor_rcd, f"output/tensor_data/{self.model_name}_decode_tensors.safetensors")
    
    def display_results(self):
        """显示测试结果表格"""
        AnalyzeTools.display_dict_as_adaptive_table(self.record, title="Test Result")
    
    def save_record(self, filename):
        """保存测试记录到文件"""
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(self.record, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {filename}")


def run_test_suite(engine, loader, target_tests, target_layers, 
                 model_name="model", test_types=None, decode_save_layers=None, decode_len=None, dataset="wiki"):
    """
    运行完整的测试套件
    
    Args:
        engine: BaseEngineSingle 或 BaseEngineMulti 实例
        loader: WikitextLoader 或 TriviaQALoader 实例
        target_tests: 测试方法列表
        target_layers: 目标层列表 [layer1, layer2]
        model_name: 模型名称
        test_types: 要运行的测试类型列表，包含以下选项：
                    - 'f1': F1分数测试
                    - 'ppl': 困惑度测试
                    - 'size': 传输大小测试
                    - 'tensor_save': 张量保存测试
                    如果为None，return
    """
    if test_types is None:
        return
    
    runner = TestRunner(engine, loader, model_name, dataset)
    engine.target_layer1 = target_layers[0]
    engine.target_layer2 = target_layers[1]
    
    for test_type in test_types:
        if test_type == 'f1':
            runner.run_f1_test(target_tests, target_layers)
        elif test_type == 'ppl':
            runner.run_ppl_test(target_tests, target_layers)
        elif test_type == 'size':
            runner.run_size_test(target_tests, target_layers)
        elif test_type == 'tensor_save':
            if decode_save_layers is None:
                runner.run_tensor_save_test()
            else:
                runner.run_tensor_save_test(save_decode=True, decode_test_layers=decode_save_layers, max_decode_len=decode_len)
        else:
            print(f"Warning: Unknown test type '{test_type}', skipping...")
    
    runner.display_results()
    return runner.record

