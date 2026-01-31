import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from tqdm import tqdm
import gc
from .edit_tools import EditTools

from torch.nn import CrossEntropyLoss

class BaseEngineMulti:
    """
    多卡版本的LLM推理引擎
    适用于Qwen-MoE等需要多GPU的模型
    """
    # 离群值检测配置
    OUTLIER_METHOD = "group_topk"  # 可选: "global_topk" 或 "group_topk"
    OUTLIER_GROUP_SIZE = 128       # 分组大小（仅在OUTLIER_METHOD="group_topk"时使用）
    OUTLIER_K = 2                  # 每个group选择的top-k数量（仅在OUTLIER_METHOD="group_topk"时使用）
    OUTLIER_RATIO = OUTLIER_K / OUTLIER_GROUP_SIZE           # 离群值比例（仅在OUTLIER_METHOD="global_topk"时使用）
    
    def __init__(self, model_path: str, device="cuda", model_type = "qwen"):
        self.device = device
        self.model_type = model_type
        print(f"Loading model from {model_path} to {self.device}...")
            
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

        # 加载模型 - 使用device_map自动分布到多卡
        if model_type == "phi":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,     
                attn_implementation="sdpa", 
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                dtype=torch.float16,     
                # attn_implementation="sdpa", 
                attn_implementation="flash_attention_2", 
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 自动探测层级结构
        self._detect_model_structure()
        self.layer_nums = len(self.layers)
        print(f"Loading {self.layer_nums} layers")
        
        # 创建EditTools，传入离群值检测配置
        self.edit_tools = EditTools(
            outlier_method=self.OUTLIER_METHOD,
            group_size=self.OUTLIER_GROUP_SIZE,
            k=self.OUTLIER_K,
            ratio=self.OUTLIER_RATIO
        )
        print(f"EditTools configured with: method={self.OUTLIER_METHOD}, group_size={self.OUTLIER_GROUP_SIZE}, k={self.OUTLIER_K}, ratio={self.OUTLIER_RATIO}")
        
        self.hooks = []
        self.record = {}

        self.target_layer1 = 0
        self.target_layer2 = 0
        self.hid1: torch.Tensor = None
        self.hid1_quant: torch.Tensor = None
        self.hid2: torch.Tensor = None
        self.hid1_size = ""
        self.hid2_size = ""
        self.hid_rcd = []
        
        # 停止符处理
        self.stop_token_ids = set()

        try:
            if self.model_type == "glm":
                self.stop_token_ids.add(self.tokenizer.eos_token_id)
                for token in ["<|endoftext|>", "<|user|>", "<|observation|>"]:
                    ids = self.tokenizer.convert_tokens_to_ids(token)
                    if isinstance(ids, int):
                        self.stop_token_ids.add(ids)
            elif self.model_type == "qwen_moe":
                self.stop_token_ids.add(self.tokenizer.eos_token_id)
                for token in ["<|im_end|>", "<|im_start|>"]:
                    ids = self.tokenizer.convert_tokens_to_ids(token)
                    if isinstance(ids, int):
                        self.stop_token_ids.add(ids)
            elif self.model_type == "llama":
                self.stop_token_ids.add(self.tokenizer.eos_token_id)
                for token in ["<|eot|>", "<|end_of_text|>","<|eom|>"]:
                    ids = self.tokenizer.convert_tokens_to_ids(token)
                    if isinstance(ids, int):
                        self.stop_token_ids.add(ids)
            elif self.model_type == "phi":
                self.stop_token_ids.add(self.tokenizer.eos_token_id)
                for token in ["<|endoftext|>", "<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]:
                    ids = self.tokenizer.convert_tokens_to_ids(token)
                    if isinstance(ids, int):
                        self.stop_token_ids.add(ids)
        except Exception as e:
            print(f"Error setting up stop tokens: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Stop Token IDs: {self.stop_token_ids}")
        
        prompt = "你是谁？，在5个字以内回答我"
        message = [
            {"role": "user", "content": prompt}
        ]

        if model_type == "glm":
            ids = self.tokenizer.apply_chat_template(message,
                                    add_generation_prompt=True, 
                                    return_tensors="pt")
        elif model_type == "qwen_moe":
            ids = self.tokenizer.apply_chat_template(message,
                                    add_generation_prompt=True, 
                                    enable_thinking=False,
                                    return_tensors="pt")
        elif model_type == "llama":
            ids = self.tokenizer.apply_chat_template(message,
                                    add_generation_prompt=True, 
                                    return_tensors="pt")
        else: return
        print(self.tokenizer.decode(ids[0], skip_special_tokens=False))
        print(self.generate(ids, 50))
        
    def _detect_model_structure(self):
        """自动探测模型的层级结构"""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # 适配 Qwen, Llama, Mistral, Yi
            self.layers = self.model.model.layers
        elif hasattr(self.model, "transformer"):
            # 适配 GLM-4, ChatGLM
            if hasattr(self.model.transformer, "encoder"):
                self.layers = self.model.transformer.encoder.layers
            elif hasattr(self.model.transformer, "layers"):
                self.layers = self.model.transformer.layers
            else:
                raise AttributeError("Could not find layers in model.transformer")
        else:
            # 备用方案
            self.layers = self.model.layers

    def _get_input_device(self):
        """获取输入层的设备"""
        return self.model.get_input_embeddings().weight.device

    def _get_hook_fn(self, layer_idx, edit_func: str = None):
        """
        构造 Hook 函数：
        1. 捕获激活值 (用于统计)
        2. 如果提供了 edit_func，则修改激活值并返回 (用于编辑)
        注意：多卡版本需要将激活值移到CPU以避免跨卡计算问题
        """
        def hook_fn(module, inputs, outputs):
            if isinstance(outputs, tuple):
                original_device = outputs[0].device
                hidden_states = outputs[0]
            else:
                original_device = outputs.device
                hidden_states = outputs
            
            # 不在单卡上运行后，需要在 CPU 保存才能运算
            if layer_idx == self.target_layer1:
                self.hid1 = hidden_states.detach().to("cpu")
            else:
                self.hid2 = hidden_states.detach().to("cpu")

            if edit_func == "INT8_Mix_Group" or edit_func == "INT8_MixGp_Group":
                modified_hidden_states, size_str = self.edit_tools.simulate_quant_int8(hidden_states, mixed_precision=True, use_group=True)
                if layer_idx == self.target_layer1:
                    self.hid1_quant = modified_hidden_states.detach().to("cpu")
            elif edit_func == "INT8_Delta":
                delta_hidden_states = self.hid2 - self.hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int8(delta_hidden_states, mixed_precision=False, use_group=False)
                modified_hidden_states = dequant_hidden_states + self.hid1
            elif edit_func == "INT8_Delta_Group":
                delta_hidden_states = self.hid2 - self.hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int8(delta_hidden_states, mixed_precision=False, use_group=True)
                modified_hidden_states = dequant_hidden_states + self.hid1
            elif edit_func == "INT8_Delta_Mix_Group" or edit_func == "INT8_Delta_MixGp_Group":
                delta_hidden_states = self.hid2 - self.hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int8(delta_hidden_states, mixed_precision=True, use_group=True)
                modified_hidden_states = dequant_hidden_states + self.hid1
            elif edit_func == "INT8_AffDelta_Group":
                aff_hid1_quant, size_bytes, slope, intercept = self.edit_tools.affine_transform(self.hid1_quant, self.hid2)
                aff_hid1 = self.edit_tools._apply_channel_affine(self.hid1, slope, intercept)
                delta_hidden_states = self.hid2 - aff_hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int8(delta_hidden_states, mixed_precision=False, use_group=True, add_bytes=size_bytes)
                modified_hidden_states = dequant_hidden_states + aff_hid1
            elif edit_func == "INT8_AffDelta_Mix_Group" or edit_func == "INT8_AffDelta_MixGp_Group":
                aff_hid1_quant, size_bytes, slope, intercept = self.edit_tools.affine_transform(self.hid1_quant, self.hid2)
                aff_hid1 = self.edit_tools._apply_channel_affine(self.hid1, slope, intercept)
                delta_hidden_states = self.hid2 - aff_hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int8(delta_hidden_states, mixed_precision=True, use_group=True, add_bytes=size_bytes)
                modified_hidden_states = dequant_hidden_states + aff_hid1
            elif edit_func == "INT8_Pertok":
                modified_hidden_states, size_str = self.edit_tools.simulate_quant_int8_per_token(hidden_states, add_bytes=0)
            elif edit_func == "INT4_Mix_Group" or edit_func == "INT4_MixGp_Group":
                modified_hidden_states, size_str = self.edit_tools.simulate_quant_int4(hidden_states, mixed_precision=True, use_group=True)
            elif edit_func == "INT4_Delta":
                delta_hidden_states = self.hid2 - self.hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int4(delta_hidden_states, mixed_precision=False, use_group=False)
                modified_hidden_states = dequant_hidden_states + self.hid1
            elif edit_func == "INT4_Delta_Group":
                delta_hidden_states = self.hid2 - self.hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int4(delta_hidden_states, mixed_precision=False, use_group=True)
                modified_hidden_states = dequant_hidden_states + self.hid1
            elif edit_func == "INT4_Delta_Mix_Group" or edit_func == "INT4_Delta_MixGp_Group":
                delta_hidden_states = self.hid2 - self.hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int4(delta_hidden_states, mixed_precision=True, use_group=True)
                modified_hidden_states = dequant_hidden_states + self.hid1
            elif edit_func == "INT4_AffDelta_Group":
                aff_hid1_quant, size_bytes, slope, intercept = self.edit_tools.affine_transform(self.hid1_quant, self.hid2)
                aff_hid1 = self.edit_tools._apply_channel_affine(self.hid1, slope, intercept)
                delta_hidden_states = self.hid2 - aff_hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int4(delta_hidden_states, mixed_precision=False, use_group=True, add_bytes=size_bytes)
                modified_hidden_states = dequant_hidden_states + aff_hid1
            elif edit_func == "INT4_AffDelta_Mix_Group" or edit_func == "INT4_AffDelta_MixGp_Group":
                aff_hid1_quant, size_bytes, slope, intercept = self.edit_tools.affine_transform(self.hid1_quant, self.hid2)
                aff_hid1 = self.edit_tools._apply_channel_affine(self.hid1, slope, intercept)
                delta_hidden_states = self.hid2 - aff_hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int4(delta_hidden_states, mixed_precision=True, use_group=True, add_bytes=size_bytes)
                modified_hidden_states = dequant_hidden_states + aff_hid1
            elif edit_func == "INT4_Pertok":
                modified_hidden_states, size_str = self.edit_tools.simulate_quant_int4_per_token(hidden_states, add_bytes=0)
            elif edit_func == "INT2_Mix_Group" or edit_func == "INT2_MixGp_Group":
                modified_hidden_states, size_str = self.edit_tools.simulate_quant_int2(hidden_states, mixed_precision=True, use_group=True)
                if layer_idx == self.target_layer1:
                    self.hid1_quant = modified_hidden_states.detach().to("cpu")
            elif edit_func == "INT2_Delta_Group":
                delta_hidden_states = self.hid2 - self.hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int2(delta_hidden_states, mixed_precision=False, use_group=True)
                modified_hidden_states = dequant_hidden_states + self.hid1
            elif edit_func == "INT2_Delta_Mix_Group" or edit_func == "INT2_Delta_MixGp_Group":
                delta_hidden_states = self.hid2 - self.hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int2(
                    delta_hidden_states, mixed_precision=True, use_group=True
                )
                modified_hidden_states = dequant_hidden_states + self.hid1
            elif edit_func == "INT2_AffDelta_Group":
                aff_hid1_quant, size_bytes, slope, intercept = self.edit_tools.affine_transform(self.hid1_quant, self.hid2)
                aff_hid1 = self.edit_tools._apply_channel_affine(self.hid1, slope, intercept)
                delta_hidden_states = self.hid2 - aff_hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int2(delta_hidden_states, mixed_precision=False, use_group=True, add_bytes=size_bytes)
                modified_hidden_states = dequant_hidden_states + aff_hid1
            elif edit_func == "INT2_AffDelta_Mix_Group" or edit_func == "INT2_AffDelta_MixGp_Group":
                aff_hid1_quant, size_bytes, slope, intercept = self.edit_tools.affine_transform(self.hid1_quant, self.hid2)
                aff_hid1 = self.edit_tools._apply_channel_affine(self.hid1, slope, intercept)
                delta_hidden_states = self.hid2 - aff_hid1_quant
                dequant_hidden_states, size_str = self.edit_tools.simulate_quant_int2(delta_hidden_states, mixed_precision=True, use_group=True, add_bytes=size_bytes)
                modified_hidden_states = dequant_hidden_states + aff_hid1
            elif edit_func == "INT2_Pertok":
                modified_hidden_states, size_str = self.edit_tools.simulate_quant_int2_per_token(hidden_states, add_bytes=0)
            elif edit_func == "FP4_Pertok":
                modified_hidden_states, size_str = self.edit_tools.simulate_quant_fp4_per_token(hidden_states, add_bytes=0)
            elif edit_func == "FP8_Pertok":
                modified_hidden_states, size_str = self.edit_tools.simulate_quant_fp8_per_token(hidden_states, add_bytes=0)
            elif edit_func == "Base":
                modified_hidden_states = None
                size_str = f"{(hidden_states.numel() * hidden_states.element_size()) / (1024 * 1024)} MB"
            else:
                print(f"[Error] Unknown edit function: {edit_func}")
                modified_hidden_states = None
                size_str = None
                
            if layer_idx == self.target_layer1:
                self.hid1_size = size_str
            else:
                self.hid2_size = size_str
            
            if modified_hidden_states is None:
                return outputs
            else:
                modified_hidden_states = modified_hidden_states.to(original_device)
            if isinstance(outputs, tuple):
                return (modified_hidden_states,) + outputs[1:]
            else:
                return modified_hidden_states
            
        return hook_fn

    def register_layer_hooks(self, target_layers_config):
        """
        注册 Hooks。
        Args:
            target_layers_config: 字典或列表。
                如果是字典: {layer_idx: edit_function}
                如果是列表: [layer_idx, ...] (仅保存，不编辑)
        """
        # 先清除旧的 hooks
        self.remove_hooks()
        
        if isinstance(target_layers_config, list):
            for layer_idx in target_layers_config:
                hook = self._get_hook_fn(layer_idx, edit_func=None)
                handle = self.layers[layer_idx].register_forward_hook(hook)
                self.hooks.append(handle)
                
        elif isinstance(target_layers_config, dict):
            for layer_idx, func in target_layers_config.items():
                hook = self._get_hook_fn(layer_idx, edit_func=func)
                handle = self.layers[layer_idx].register_forward_hook(hook)
                self.hooks.append(handle)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def step_inference(self, input_ids, past_key_values=None):
        """
        执行单步推理
        """
        # 清空上一各 token 的临时存储
        self.hid1 = None
        self.hid2 = None 
        
        first_layer_device = self._get_input_device()
        input_ids = input_ids.to(first_layer_device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, 
                past_key_values=past_key_values,
                use_cache=True
            )
        
        return outputs.logits, outputs.past_key_values

    def generate(self, prompt_or_ids, max_new_tokens=20, enable_hid_rcd=False):
        """
        生成函数
        Args:
            prompt_or_ids: 提示词字符串或 Tensor
            max_new_tokens: 生成长度
            layers_config: 传递给 register_layer_hooks 的配置
        """
        if isinstance(prompt_or_ids, str):
            inputs = self.tokenizer(prompt_or_ids, return_tensors="pt")
            input_ids = inputs.input_ids
        else:
            input_ids = prompt_or_ids
            
        self.hid_rcd = []
        generated_ids = []
        past_key_values = None
        if self.model_type == "phi":
            past_key_values = DynamicCache()
        curr_input = input_ids 
            
        for i in range(max_new_tokens):
            logits, past_key_values = self.step_inference(curr_input, past_key_values)
            
            # 贪婪采样
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            token_val = next_token_id.item()
            
            # print(f"{self.tokenizer.decode([token_val], skip_special_tokens=False)}, {token_val}")
            
            if token_val in self.stop_token_ids:
                print(f" [STOP] {token_val}", end="")
                break
            
            generated_ids.append(token_val)
                
            curr_input = next_token_id
            
            if enable_hid_rcd:
                if self.hid1 is not None and self.hid2 is not None:
                    self.hid_rcd.append({
                        "layer1_hid": self.hid1.cpu(),
                        "layer2_hid": self.hid2.cpu(),
                    })
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=False)

    def calculate_perplexity(self, input_ids, stride=512, max_length=None):
        if max_length is None:
            max_length = getattr(self.model.config, "max_position_embeddings", 2048)
        
        effective_max_len = max_length
        if self.model_type == "glm":
            print("PPL test add prefix for glm")
            prefix_ids = [151331, 151333]
            prefix_len = len(prefix_ids)
            effective_max_len = max_length - prefix_len
        
        print(f"\n[Evaluation] Calculating PPL (Stride={stride}, Context={max_length})...")
        
        if input_ids.ndim == 2:
            input_ids = input_ids.squeeze(0)
            
        seq_len = input_ids.size(0)
        nlls = []
        prev_end_loc = 0
        
        self.model.eval()
        
        # 实例化 Loss 函数
        loss_fct = CrossEntropyLoss(reduction='none') 

        for begin_loc in tqdm(range(0, seq_len, stride), desc="PPL Progress"):
            end_loc = min(begin_loc + effective_max_len, seq_len)
            trg_len = end_loc - prev_end_loc
            
            if self.model_type == "glm":
                chunk_list = input_ids[begin_loc:end_loc].tolist()
                chunk_list = prefix_ids + chunk_list
                input_ids_chunk = torch.tensor([chunk_list], device=self._get_input_device(), dtype=torch.long)
            else:
                input_ids_chunk = input_ids[begin_loc:end_loc].unsqueeze(0)
            
            input_ids_chunk = input_ids_chunk.to(self._get_input_device())
            target_ids = input_ids_chunk.clone()
            
            # Mask 掉不需要计算 loss 的部分（即 Context 部分）
            # 注意：这里的 target_ids 还在 input_device 上
            target_ids[:, :-trg_len] = -100 

            with torch.no_grad():
                # 1. 这里的改动：不要传入 labels，只传 input_ids，只拿 logits
                outputs = self.model(input_ids_chunk)
                logits = outputs.logits

                # 2. Causal LM 的标准操作：Shift Logits 和 Labels
                # Logits 预测的是下一个 token，所以我们要用第 N 个 logit 去对齐第 N+1 个 label
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()

                # 3. 关键修复：强制将 labels 移动到 logits 所在的设备 (cuda:1)
                shift_labels = shift_labels.to(shift_logits.device)

                # 4. 手动计算 CrossEntropy
                # view(-1, ...) 是为了拉平 batch 和 sequence 维度以适配 Loss 函数
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).float(), shift_labels.view(-1))
                
                # 5. 求和得到该窗口的总 NLL
                # 因为用了 reduction='none'，loss 是逐 token 的，我们需要对非 -100 的部分求和
                # 或者更简单的：如果用 reduction='mean' (默认)，则 neg_log_likelihood = loss * trg_len
                # 这里为了稳妥，建议用 sum:
                neg_log_likelihood = loss.sum() 

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        # 这里的总 NLL 已经是所有 token loss 的和，直接除以总长度即可
        total_nll = torch.stack(nlls).sum()
        ppl = torch.exp(total_nll / seq_len)
        return ppl.item()
