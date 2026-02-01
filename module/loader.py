from transformers import AutoTokenizer
from datasets import load_dataset
import torch

class WikitextLoader:
    def __init__(self, tokenizer, file_path=None, seq_len=512, split="test", model="qwen", device=None):
        """
        Wikitext数据集加载器
        
        Args:
            tokenizer: 分词器
            file_path: 本地文件路径（如果提供则从本地加载）
            seq_len: 序列长度
            split: 数据集分割
            model: 模型类型（用于特殊处理，如"glm"）
            device: 设备（可选）
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = device
        self.model_type = model
        
        # 加载数据集
        if file_path and file_path.endswith(".parquet"):
            dataset_dict = load_dataset("parquet", data_files={split: file_path})
            self.dataset = dataset_dict[split]
        elif file_path and file_path.endswith(".txt"):
            dataset_dict = load_dataset("text", data_files={split: file_path})
            self.dataset = dataset_dict[split]
        else:
            self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # 根据模型类型处理数据
        if model == "glm":
            self._prepare_glm_data()
        else:
            self._prepare_standard_data()

    def _prepare_glm_data(self):
        """为GLM模型准备数据（添加特殊token）"""
        print(f"[Data] Joining all text lines for GLM...")
        self.full_text = "\n\n".join([text for text in self.dataset["text"] if text.strip()])
        
        print(f"[Data] Tokenizing full text to get input_ids...")
        with torch.no_grad():
            full_inputs = self.tokenizer(self.full_text, add_special_tokens=False, return_tensors="pt")
            self.input_ids = full_inputs.input_ids
        
        if self.device:
            self.input_ids = self.input_ids.to(self.device)
        
        self.total_tokens = self.input_ids.size(1)
        print(f"[Data] Ready. Total tokens: {self.total_tokens}")

    def _prepare_standard_data(self):
        """为标准模型准备数据"""
        all_ids = []
        for text in self.dataset["text"]:
            if text.strip():
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                all_ids.extend(ids)
        
        self.input_ids = torch.tensor(all_ids, dtype=torch.long).unsqueeze(0)
        self.total_tokens = self.input_ids.size(1)
        
        if self.device:
            self.input_ids = self.input_ids.to(self.device)
        
        print(f"[Data] Ready. Total tokens: {self.total_tokens}")

    def get_batches(self):
        """
        生成器：按 seq_len 产出 input_ids
        
        对于GLM模型，会重新编码以添加特殊token
        """
        if self.model_type == "glm":
            effective_len = self.seq_len - 2  # GLM需要2个前缀token
            num_batches = self.total_tokens // effective_len
            
            for i in range(num_batches):
                start = i * effective_len
                end = start + effective_len
                
                chunk_ids = self.input_ids[0, start:end].tolist()
                text_chunk = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
                
                final_inputs = self.tokenizer(
                    text_chunk, 
                    add_special_tokens=True, 
                    return_tensors="pt"
                )
                
                if self.device:
                    final_inputs = final_inputs.to(self.device)
                    
                yield final_inputs.input_ids
        else:
            num_batches = self.total_tokens // self.seq_len
            
            for i in range(num_batches):
                start = i * self.seq_len
                end = start + self.seq_len
                yield self.input_ids[:, start:end]

class TriviaQALoader:
    def __init__(self, tokenizer, file_path, use_context=False, max_samples=None, model="qwen", start_index=0):
        """
        TriviaQA数据集加载器
        
        Args:
            tokenizer: 分词器
            file_path: 本地parquet文件路径
            use_context: 是否在Prompt中加入维基百科上下文
            max_samples: 仅加载前N条数据
            start_index: 从第几个问题开始（0-based）
        """
        self.tokenizer = tokenizer
        self.use_context = use_context
        self.model=model
        self.start_index = start_index

        print(f"[Data] Loading local file: {file_path}")
        
        self.dataset = load_dataset(
            "parquet", 
            data_files={"validation": file_path}, 
            split="validation"
        )
        if self.start_index:
            print(f"[Data] Skipping first {self.start_index} samples.")
            self.dataset = self.dataset.select(range(self.start_index, len(self.dataset)))
        
        if max_samples:
            print(f"[Data] Truncating to {max_samples} samples.")
            self.dataset = self.dataset.select(range(max_samples))
            
        print(f"[Data] Ready. Total samples loaded: {len(self.dataset)}")

    def _format_prompt(self, question, context=""):
        """
        构造Prompt格式
        """
        if self.use_context and context:
            return f"Read the following context and answer the question.Give only the final answer.Do not include any explanation, background, or additional text.\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            return f"Please answer the following question.Give only the final answer.Do not include any explanation, background, or additional text.\nQuestion:{question}\nAnswer:"

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for sample in self.dataset:
            question = sample['question']
            question_id = sample['question_id']
            
            ground_truths = sample['answer']['aliases']
            
            context_text = ""
            if self.use_context:
                if sample.get('entity_pages') and len(sample['entity_pages']['wiki_context']) > 0:
                    context_text = sample['entity_pages']['wiki_context'][0]
                    context_text = context_text[:8192] 

            prompt_text = self._format_prompt(question, context_text)
            
            if self.model == "glm":
                message = [
                    {"role":"user", "content": prompt_text}
                ]
                input_ids = self.tokenizer.apply_chat_template(message,
                                add_generation_prompt=True, 
                                return_tensors="pt")
            elif self.model == "qwen_moe":
                message = [
                    {"role": "user", "content": prompt_text}
                ]
                input_ids = self.tokenizer.apply_chat_template(message,
                                add_generation_prompt=True, 
                                enable_thinking=False,
                                return_tensors="pt")
            elif self.model == "llama":
                message = [
                    {"role":"user", "content": prompt_text}
                ]
                input_ids = self.tokenizer.apply_chat_template(message,
                                add_generation_prompt=True, 
                                return_tensors="pt")
            elif self.model == "phi":
                message = [
                    {"role":"user", "content": prompt_text}
                ]
                input_ids = self.tokenizer.apply_chat_template(message,
                                add_generation_prompt=True, 
                                return_tensors="pt")
            else:
                print(f"Unkown model type: {self.model}, using default tokenization.")
                input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids

            yield {
                "id": question_id,
                "input_ids": input_ids,
                "question": prompt_text,
                "ground_truths": ground_truths
            }

    def get_batches(self):
        """
        生成器：按样本产出 input_ids
        """
        for item in self:
            yield item["input_ids"]


class NarrativeQALoader:
    def __init__(self, tokenizer, file_path, use_context=False, max_samples=None, model="qwen", start_index=0):
        """
        
        Args:
            tokenizer: 分词器
            file_path: 本地parquet文件路径
            use_context: 是否在Prompt中加入维基百科上下文
            max_samples: 仅加载前N条数据
            start_index: 从第几个问题开始（0-based）
        """
        self.tokenizer = tokenizer
        self.use_context = use_context
        self.model=model
        self.start_index = start_index
        print(f"[Data] Loading local file: {file_path}")
        
        self.dataset = load_dataset(
            "parquet", 
            data_files={"test": file_path}, 
            split="test"
        )
        if self.start_index:
            print(f"[Data] Skipping first {self.start_index} samples.")
            self.dataset = self.dataset.select(range(self.start_index, len(self.dataset)))
        
        if max_samples:
            print(f"[Data] Truncating to {max_samples} samples.")
            self.dataset = self.dataset.select(range(max_samples))
            
        print(f"[Data] Ready. Total samples loaded: {len(self.dataset)}")

    def _format_prompt(self, question, context=""):
        """
        构造Prompt格式
        """
        if self.use_context and context:
            return f"Read the following context and answer the question.Give only the final answer.Do not include any explanation, background, or additional text.\nContext: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            return f"Please answer the following question.Give only the final answer.Do not include any explanation, background, or additional text.\nQuestion:{question}\nAnswer:"

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        idx = self.start_index
        for sample in self.dataset:
            question = sample['question']['text']
            question_id = idx
            
            ground_truths = [s['text'] for s in sample['answers']]

            context_text = ""
            if self.use_context:
                if sample.get('document') and len(sample['document']['summary']['text']) > 0:
                    context_text = sample['document']['summary']['text']
                    context_text = context_text[:8192] 

            prompt_text = self._format_prompt(question, context_text)
            
            if self.model == "glm":
                message = [
                    {"role":"user", "content": prompt_text}
                ]
                input_ids = self.tokenizer.apply_chat_template(message,
                                add_generation_prompt=True, 
                                return_tensors="pt")
            elif self.model == "qwen_moe":
                message = [
                    {"role": "user", "content": prompt_text}
                ]
                input_ids = self.tokenizer.apply_chat_template(message,
                                add_generation_prompt=True, 
                                enable_thinking=False,
                                return_tensors="pt")
            elif self.model == "llama":
                message = [
                    {"role":"user", "content": prompt_text}
                ]
                input_ids = self.tokenizer.apply_chat_template(message,
                                add_generation_prompt=True, 
                                return_tensors="pt")
            elif self.model == "phi":
                message = [
                    {"role":"user", "content": prompt_text}
                ]
                input_ids = self.tokenizer.apply_chat_template(message,
                                add_generation_prompt=True, 
                                return_tensors="pt")
            else:
                print(f"Unkown model type: {self.model}, using default tokenization.")
                input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids
                
            idx+=1

            yield {
                "id": question_id,
                "input_ids": input_ids,
                "question": prompt_text,
                "ground_truths": ground_truths
            }

    def get_batches(self):
        """
        生成器：按样本产出 input_ids
        """
        for item in self:
            yield item["input_ids"]


class LongChatLoader:
    def __init__(self, tokenizer, file_path, input_length=1024, split="test", device=None):
        """
        LongChat数据集加载器

        Args:
            tokenizer: 分词器
            file_path: 本地jsonl文件路径
            input_length: 每次yield的文本长度（按token计）
            split: 数据集分割名称（jsonl通常只有一个split，可随意指定）
            device: 设备（可选）
        """
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.device = device

        print(f"[Data] Loading local file: {file_path}")
        self.dataset = load_dataset(
            "json",
            data_files={split: file_path},
            split=split
        )
        print(f"[Data] Ready. Total samples loaded: {len(self.dataset)}")

    def get_batches(self):
        """
        生成器：每次yield一个样本的input的input_length长度文本，并转换为input_ids
        """
        for sample in self.dataset:
            text = sample.get("input", "")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = ids[: self.input_length]
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            if self.device:
                input_ids = input_ids.to(self.device)
            yield input_ids

