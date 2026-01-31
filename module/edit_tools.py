import torch

class EditTools:
    def __init__(self, outlier_method="global_topk", group_size=128, k=8, ratio=0.01):
        """
        Args:
            outlier_method: 离群值检测方法，可选 "global_topk" (基于ratio的全局topk) 或 "group_topk" (基于k的分组topk)
            group_size: 分组大小（仅在outlier_method="group_topk"时使用）
            k: 每个group选择的top-k数量（仅在outlier_method="group_topk"时使用）
            ratio: 离群值比例（仅在outlier_method="global_topk"时使用）
        """
        self.outlier_method = outlier_method
        self.group_size = group_size
        self.k = k
        self.ratio = ratio
    
    def _compute_outlier_threshold(self, tensor, threshold=3.0):
        data = tensor.float()
        rms = torch.sqrt(torch.mean(data**2)) # 均方根RMS
        return threshold * rms.item()
    
    def extract_outliers_group_topk(self, tensor, group_size=128, k=8):
        """
        Group Top K - Same Grouping: 使用与分组量化完全相同的分组方式
        按 token(行) 分组，只在 hidden 维度 padding + group，与 quantize_int8/int4_generic 对齐
        """
        x = tensor.clone().float()
        original_shape = x.shape
        
        # 步骤1：reshape 到 [seq_len, hidden]（与 quantize 中 x_flat_view 对齐）
        x_flat_view = x.reshape(-1, original_shape[-1])
        seq_len, hidden = x_flat_view.shape
        
        # 步骤2：只在 hidden 维度 padding（与 quantize 中 pad_len 对齐）
        pad_len = (group_size - hidden % group_size) % group_size
        if pad_len > 0:
            x_padded = torch.nn.functional.pad(x_flat_view, (0, pad_len))
        else:
            x_padded = x_flat_view
        
        # 步骤3：分组 [seq_len, num_groups, group_size]（与 quantize 中 x_groups 对齐）
        num_groups = x_padded.shape[1] // group_size
        x_groups = x_padded.view(seq_len, num_groups, group_size)
        x_abs = x_groups.abs()
        
        # 步骤4：在每个 group 内选 top-k（dim=-1，即 group_size 维度）
        topk_vals, topk_inds = torch.topk(x_abs, k=min(k, group_size), dim=-1)
        
        # 步骤5：创建 mask（与 quantize 中 group mask 结构一致）
        group_mask = torch.zeros_like(x_groups, dtype=torch.bool)
        group_mask.scatter_(-1, topk_inds, True)
        
        # 步骤6：flatten 并去除 padding（与 quantize 中 final mask 对齐）
        mask_padded = group_mask.view(seq_len, -1)
        if pad_len > 0:
            mask_flat = mask_padded[:, :-pad_len]
        else:
            mask_flat = mask_padded
        
        final_mask = mask_flat.view(original_shape)
        
        return final_mask

    def extract_outliers_global_topk(self, tensor, ratio=0.005):
        x = tensor.clone()
        total_elements = x.numel()
        mask = torch.zeros_like(x, dtype=torch.bool)

        if ratio > 0:
            x_flat_abs = x.abs().flatten()
            k = int(total_elements * ratio)
            if k > 0:
                threshold_val = torch.kthvalue(x_flat_abs, total_elements - k).values
                mask = x.abs() > threshold_val

        return mask

    def quantize_fp16_int8(self, tensor, group_size=128, outlier_ratio=0.005, mixed_precision=True, use_grouping=True, add_bytes=0):
        """
        INT8 量化函数：支持混合精度(百分位法)、分组量化。
        
        Args:
            tensor (torch.Tensor): 输入张量 (FP16)。
            group_size (int): 分组大小 (仅在 use_grouping=True 时生效)。
            outlier_ratio (float): 离群值比例，例如 0.005 代表去除最大的 0.5% 数据。
            mixed_precision (bool): 是否保留离群值。
            use_grouping (bool): 是否使用分组量化。False 则退化为 Per-Token 量化。
            
        Returns:
            dict: 量化数据包。
            str: 数据大小字符串。
        """
        x = tensor.clone().float()
        original_shape = x.shape
        total_elements = x.numel()
        
        if mixed_precision:
            # 使用配置的离群值检测方法
            if self.outlier_method == "group_topk":
                mask = self.extract_outliers_group_topk(tensor, self.group_size, self.k)
            elif self.outlier_method == "global_topk":
                mask = self.extract_outliers_global_topk(tensor, self.ratio)
            else:
                raise ValueError(f"Unknown outlier_method: {self.outlier_method}. Use 'group_topk' or 'global_topk'.")
            
            k = mask.sum().item()
            if k > 0:
                outlier_values = x[mask].to(torch.float16)
                outlier_indices = torch.nonzero(mask, as_tuple=False)
            else:
                outlier_values = torch.empty(0)
                outlier_indices = torch.empty(0)
        else:
            mask = torch.zeros_like(x, dtype=torch.bool)
            k = 0
            outlier_values = torch.empty(0)
            outlier_indices = torch.empty(0)
            outlier_size = 0

        x_flat_view = x.reshape(-1, original_shape[-1])
        mask_flat_view = mask.reshape(-1, original_shape[-1])
        seq_len, hidden = x_flat_view.shape
        
        if use_grouping:
            effective_group_size = group_size
        else:
            effective_group_size = hidden

        pad_len = (effective_group_size - hidden % effective_group_size) % effective_group_size
        
        if pad_len > 0:
            x_padded = torch.nn.functional.pad(x_flat_view, (0, pad_len))
            mask_padded = torch.nn.functional.pad(mask_flat_view, (0, pad_len), value=False)
        else:
            x_padded = x_flat_view
            mask_padded = mask_flat_view
            
        padded_hidden = x_padded.shape[1]
        num_groups = padded_hidden // effective_group_size
        
        x_groups = x_padded.view(seq_len, num_groups, effective_group_size)
        mask_groups = mask_padded.view(seq_len, num_groups, effective_group_size)
        
        INF = torch.finfo(torch.float32).max / 2
        x_for_min = x_groups.clone()
        x_for_max = x_groups.clone()
        
        if mixed_precision:
            x_for_min[mask_groups] = INF
            x_for_max[mask_groups] = -INF
        
        # 计算 Min/Max
        min_val = x_for_min.min(dim=-1, keepdim=True)[0]
        max_val = x_for_max.max(dim=-1, keepdim=True)[0]
        
        if mixed_precision:
            all_outlier = mask_groups.all(dim=-1, keepdim=True)
            min_val = torch.where(all_outlier, torch.zeros_like(min_val), min_val)
            max_val = torch.where(all_outlier, torch.ones_like(max_val), max_val)
            
        range_val = (max_val - min_val).clamp(min=1e-6)
        base_bits = 8
        max_q_val = 2**base_bits - 1
        scale = range_val / max_q_val
        
        x_q = ((x_groups - min_val) / scale).round().clamp(0, max_q_val)
        x_q = x_q.to(torch.uint8)

        # Outlier 部分大小
        if mixed_precision and k > 0:
            outlier_values_bytes = k * 2 
            coo_indices_bytes = k * 4    
            bitmap_bytes = (total_elements + 7) // 8
            outlier_indices_bytes = min(coo_indices_bytes, bitmap_bytes)
            outlier_size = outlier_values_bytes + outlier_indices_bytes
        else:
            outlier_size = 0
        
        # Body Payload (INT8 = 1 byte per element)
        body_payload_bytes = total_elements * 1.0
        
        # Metadata (Min/Scale)
        num_actual_groups = min_val.numel() 
        body_metadata_bytes = num_actual_groups * 4 
        
        total_bytes = outlier_size + body_payload_bytes + body_metadata_bytes + add_bytes
        
        if total_bytes < 1024:
            size_str = f"{total_bytes:.0f} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes/1024:.2f} KB"
        else:
            size_str = f"{total_bytes/(1024**2):.2f} MB"

        quantized_data = {
            "q_data": x_q,
            "scales": scale.half(),
            "min_vals": min_val.half(),
            "outlier_indices": outlier_indices,
            "outlier_values": outlier_values,
            "original_shape": original_shape,
            "pad_len": pad_len,
            "mixed_precision": mixed_precision,
            "effective_group_size": effective_group_size 
        }
        
        return quantized_data, size_str

    def dequantize_int8_fp16(self, quantized_data):
        """
        解量化函数：逻辑保持不变，依赖 quantized_data 中的维度信息自动适配
        """
        x_q = quantized_data["q_data"]
        scale = quantized_data["scales"].float()
        min_val = quantized_data["min_vals"].float()
        outlier_indices = quantized_data["outlier_indices"]
        outlier_values = quantized_data["outlier_values"]
        original_shape = quantized_data["original_shape"]
        pad_len = quantized_data["pad_len"]
        mixed_precision = quantized_data["mixed_precision"]
        
        x_recon_groups = x_q.float() * scale + min_val
        
        seq_len = x_recon_groups.shape[0]
        padded_hidden = x_recon_groups.shape[1] * x_recon_groups.shape[2]
        
        x_recon_flat = x_recon_groups.view(seq_len, padded_hidden)
        
        if pad_len > 0:
            actual_hidden = padded_hidden - pad_len
            x_recon_flat = x_recon_flat[:, :actual_hidden]
        else:
            x_recon_flat = x_recon_flat # No padding to remove
            
        x_recon = x_recon_flat.reshape(original_shape)
        
        if mixed_precision and outlier_indices.numel() > 0:
            indices_tuple = tuple(outlier_indices.t())
            x_recon.index_put_(indices_tuple, outlier_values.float())
            
        return x_recon.to(torch.float16)
    
    def quantize_fp16_int4(self, tensor, group_size=128, outlier_ratio=0.01, mixed_precision=True, use_grouping=True, add_bytes=0):
        """
        INT4 量化函数：支持混合精度(百分位法)、分组量化、位打包。
        Args:
            outlier_ratio (float): 离群值比例，例如 0.005 代表去除最大的 0.5% 数据。
        """
        x = tensor.clone().float()
        original_shape = x.shape
        total_elements = x.numel()
        
        if mixed_precision:
            # 使用配置的离群值检测方法
            if self.outlier_method == "group_topk":
                mask = self.extract_outliers_group_topk(tensor, self.group_size, self.k)
            elif self.outlier_method == "global_topk":
                mask = self.extract_outliers_global_topk(tensor, self.ratio)
            else:
                raise ValueError(f"Unknown outlier_method: {self.outlier_method}. Use 'group_topk' or 'global_topk'.")
            
            k = mask.sum().item()
            if k > 0:
                outlier_values = x[mask].to(torch.float16)
                outlier_indices = torch.nonzero(mask, as_tuple=False)
            else:
                outlier_values = torch.empty(0)
                outlier_indices = torch.empty(0)
        else:
            mask = torch.zeros_like(x, dtype=torch.bool)
            k = 0
            outlier_values = torch.empty(0)
            outlier_indices = torch.empty(0)
            outlier_size = 0

        x_flat_view = x.reshape(-1, original_shape[-1])
        mask_flat_view = mask.reshape(-1, original_shape[-1])
        seq_len, hidden = x_flat_view.shape
        
        if use_grouping:
            effective_group_size = group_size
        else:
            effective_group_size = hidden

        if effective_group_size % 2 != 0:
             effective_group_size += 1
             
        pad_len = (effective_group_size - hidden % effective_group_size) % effective_group_size
        
        if pad_len > 0:
            x_padded = torch.nn.functional.pad(x_flat_view, (0, pad_len))
            mask_padded = torch.nn.functional.pad(mask_flat_view, (0, pad_len), value=False)
        else:
            x_padded = x_flat_view
            mask_padded = mask_flat_view
            
        padded_hidden = x_padded.shape[1]
        num_groups = padded_hidden // effective_group_size
        
        x_groups = x_padded.view(seq_len, num_groups, effective_group_size)
        mask_groups = mask_padded.view(seq_len, num_groups, effective_group_size)
        
        INF = torch.finfo(torch.float32).max / 2
        x_for_min = x_groups.clone()
        x_for_max = x_groups.clone()
        
        if mixed_precision:
            # 排除离群值，不让它们影响 Min/Max 的计算
            x_for_min[mask_groups] = INF
            x_for_max[mask_groups] = -INF
        
        min_val = x_for_min.min(dim=-1, keepdim=True)[0]
        max_val = x_for_max.max(dim=-1, keepdim=True)[0]
        
        if mixed_precision:
            all_outlier = mask_groups.all(dim=-1, keepdim=True)
            min_val = torch.where(all_outlier, torch.zeros_like(min_val), min_val)
            max_val = torch.where(all_outlier, torch.ones_like(max_val), max_val)
            
        range_val = (max_val - min_val).clamp(min=1e-6)
        
        # INT4 配置
        base_bits = 4
        max_q_val = 2**base_bits - 1  # 15
        scale = range_val / max_q_val
        
        x_q = ((x_groups - min_val) / scale).round().clamp(0, max_q_val)
        x_q = x_q.to(torch.uint8)
        
        # 将两个 4-bit 数塞进一个 8-bit 数
        # 取偶数位作为高 4 位，奇数位作为低 4 位
        x_q_high = x_q[..., 0::2]
        x_q_low = x_q[..., 1::2]
        
        # Packing: (High << 4) | Low
        x_packed = (x_q_high << 4) | x_q_low

        # Outlier 部分大小
        if mixed_precision and k > 0:
            outlier_values_bytes = k * 2 
            coo_indices_bytes = k * 4    
            bitmap_bytes = (total_elements + 7) // 8
            outlier_indices_bytes = min(coo_indices_bytes, bitmap_bytes)
            outlier_size = outlier_values_bytes + outlier_indices_bytes
        else:
            outlier_size = 0
        
        # Payload: 4 bits per element -> total / 2 bytes
        body_payload_bytes = total_elements * 0.5 
        
        # Metadata
        num_actual_groups = min_val.numel()
        body_metadata_bytes = num_actual_groups * 4 
        
        total_bytes = outlier_size + body_payload_bytes + body_metadata_bytes + add_bytes
        
        if total_bytes < 1024:
            size_str = f"{total_bytes:.0f} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes/1024:.2f} KB"
        else:
            size_str = f"{total_bytes/(1024**2):.2f} MB"

        quantized_data = {
            "q_data": x_packed,
            "scales": scale.half(),
            "min_vals": min_val.half(),
            "outlier_indices": outlier_indices,
            "outlier_values": outlier_values,
            "original_shape": original_shape,
            "pad_len": pad_len,
            "mixed_precision": mixed_precision,
            "effective_group_size": effective_group_size
        }
        
        return quantized_data, size_str

    def dequantize_int4_fp16(self, quantized_data):
        """
        INT4 解量化函数：先解包再反量化。
        """
        x_packed = quantized_data["q_data"]
        scale = quantized_data["scales"].float()
        min_val = quantized_data["min_vals"].float()
        outlier_indices = quantized_data["outlier_indices"]
        outlier_values = quantized_data["outlier_values"]
        original_shape = quantized_data["original_shape"]
        pad_len = quantized_data["pad_len"]
        mixed_precision = quantized_data["mixed_precision"]
        
        # x_packed shape: [Seq, Num_Groups, Group_Size // 2]
        
        # 提取高 4 位 (右移4位后取低4位)
        x_q_high = (x_packed >> 4) & 0x0F
        # 提取低 4 位 (直接取低4位)
        x_q_low = x_packed & 0x0F
        
        # 重新交织 (Interleave) 以恢复顺序
        # 方法：Stack 后 reshape
        # 假设 packed 是 [..., L/2]，我们 stack 变成 [..., L/2, 2]，然后 flatten 变成 [..., L]
        x_q_stacked = torch.stack((x_q_high, x_q_low), dim=-1)
        x_q = x_q_stacked.flatten(start_dim=-2) 
        
        x_recon_groups = x_q.float() * scale + min_val
        
        seq_len = x_recon_groups.shape[0]
        padded_hidden = x_recon_groups.shape[1] * x_recon_groups.shape[2]
        
        x_recon_flat = x_recon_groups.view(seq_len, padded_hidden)
        
        if pad_len > 0:
            actual_hidden = padded_hidden - pad_len
            x_recon_flat = x_recon_flat[:, :actual_hidden]
        
        x_recon = x_recon_flat.reshape(original_shape)
        
        if mixed_precision and outlier_indices.numel() > 0:
            indices_tuple = tuple(outlier_indices.t())
            x_recon.index_put_(indices_tuple, outlier_values.float())
            
        return x_recon.to(torch.float16)
    
    def quantize_fp16_int2(self, tensor, group_size=128, outlier_ratio=0.01, mixed_precision=True, use_grouping=True, add_bytes=0):
        """
        INT2 量化函数：支持混合精度、分组量化、4-to-1 位打包。
        """
        x = tensor.clone().float()
        original_shape = x.shape
        total_elements = x.numel()
        
        # 1. 离群值处理 (Outlier Detection)
        if mixed_precision:
            if self.outlier_method == "group_topk":
                mask = self.extract_outliers_group_topk(tensor, self.group_size, self.k)
            elif self.outlier_method == "global_topk":
                mask = self.extract_outliers_global_topk(tensor, self.ratio)
            else:
                raise ValueError(f"Unknown outlier_method: {self.outlier_method}")
            
            k = mask.sum().item()
            if k > 0:
                outlier_values = x[mask].to(torch.float16)
                outlier_indices = torch.nonzero(mask, as_tuple=False)
            else:
                outlier_values = torch.empty(0)
                outlier_indices = torch.empty(0)
        else:
            mask = torch.zeros_like(x, dtype=torch.bool)
            k = 0
            outlier_values = torch.empty(0)
            outlier_indices = torch.empty(0)

        # 2. 准备分组与 Padding
        x_flat_view = x.reshape(-1, original_shape[-1])
        mask_flat_view = mask.reshape(-1, original_shape[-1])
        seq_len, hidden = x_flat_view.shape
        
        effective_group_size = group_size if use_grouping else hidden
        # INT2 打包要求每组大小必须是 4 的倍数
        if effective_group_size % 4 != 0:
             effective_group_size += (4 - effective_group_size % 4)
             
        pad_len = (effective_group_size - hidden % effective_group_size) % effective_group_size
        
        if pad_len > 0:
            x_padded = torch.nn.functional.pad(x_flat_view, (0, pad_len))
            mask_padded = torch.nn.functional.pad(mask_flat_view, (0, pad_len), value=False)
        else:
            x_padded = x_flat_view
            mask_padded = mask_flat_view
            
        num_groups = x_padded.shape[1] // effective_group_size
        x_groups = x_padded.view(seq_len, num_groups, effective_group_size)
        mask_groups = mask_padded.view(seq_len, num_groups, effective_group_size)
        
        # 3. 计算 Min/Max (排除离群值)
        INF = torch.finfo(torch.float32).max / 2
        x_for_min, x_for_max = x_groups.clone(), x_groups.clone()
        if mixed_precision:
            x_for_min[mask_groups], x_for_max[mask_groups] = INF, -INF
        
        min_val = x_for_min.min(dim=-1, keepdim=True)[0]
        max_val = x_for_max.max(dim=-1, keepdim=True)[0]
        
        range_val = (max_val - min_val).clamp(min=1e-6)
        max_q_val = 3  # 2^2 - 1
        scale = range_val / max_q_val
        
        # 4. 量化与位打包 (Packing)
        x_q = ((x_groups - min_val) / scale).round().clamp(0, max_q_val).to(torch.uint8)
        
        # 将 4 个 2-bit 数打包进一个 uint8
        # x_q shape: [Seq, Num_Groups, Group_Size]
        v0 = x_q[..., 0::4] << 6
        v1 = x_q[..., 1::4] << 4
        v2 = x_q[..., 2::4] << 2
        v3 = x_q[..., 3::4]
        x_packed = v0 | v1 | v2 | v3

        # 5. 传输大小计算
        outlier_size = 0
        if mixed_precision and k > 0:
            outlier_size = (k * 2) + min(k * 4, (total_elements + 7) // 8)
        
        body_payload_bytes = total_elements * 0.25 # 每个元素 2 bits
        body_metadata_bytes = min_val.numel() * 4 # Min & Scale 为 FP16
        total_bytes = outlier_size + body_payload_bytes + body_metadata_bytes + add_bytes
        
        size_str = f"{total_bytes/1024:.2f} KB" if total_bytes < 1024**2 else f"{total_bytes/(1024**2):.2f} MB"

        return {
            "q_data": x_packed,
            "scales": scale.half(),
            "min_vals": min_val.half(),
            "outlier_indices": outlier_indices,
            "outlier_values": outlier_values,
            "original_shape": original_shape,
            "pad_len": pad_len,
            "mixed_precision": mixed_precision,
            "effective_group_size": effective_group_size
        }, size_str
        
    def dequantize_int2_fp16(self, quantized_data):
        """
        INT2 解量化函数：先解包再反量化。
        """
        x_packed = quantized_data["q_data"]
        scale, min_val = quantized_data["scales"].float(), quantized_data["min_vals"].float()
        outlier_indices = quantized_data["outlier_indices"]
        outlier_values = quantized_data["outlier_values"]
        original_shape = quantized_data["original_shape"]
        pad_len, mixed_precision = quantized_data["pad_len"], quantized_data["mixed_precision"]
        
        # 解包 4 个 2-bit 值
        v0 = (x_packed >> 6) & 0x03
        v1 = (x_packed >> 4) & 0x03
        v2 = (x_packed >> 2) & 0x03
        v3 = x_packed & 0x03
        
        # 恢复顺序
        x_q = torch.stack((v0, v1, v2, v3), dim=-1).flatten(start_dim=-2)
        
        # 反量化
        x_recon_groups = x_q.float() * scale + min_val
        x_recon_flat = x_recon_groups.view(x_recon_groups.shape[0], -1)
        
        # 移除 Padding 并恢复形状
        if pad_len > 0:
            x_recon_flat = x_recon_flat[:, :-pad_len]
        x_recon = x_recon_flat.reshape(original_shape)
        
        # 还原离群值
        if mixed_precision and outlier_indices.numel() > 0:
            x_recon.index_put_(tuple(outlier_indices.t()), outlier_values.float())
            
        return x_recon.to(torch.float16)

    def extract_outliers(self, tensor, ratio=0.01):
        x = tensor.clone()
        total_elements = x.numel()
        
        k = int(total_elements * ratio)
        
        if k > 0:
            # 展平求阈值，保持逻辑一致
            x_flat_abs = x.abs().view(-1)
            threshold = torch.kthvalue(x_flat_abs, total_elements - k).values
            
            mask = x.abs() > threshold
            
            # 重新统计精确的 k (因为可能有多个值等于阈值)
            k = mask.sum().item()
            
            # 提取值和索引
            outlier_values = x[mask].to(torch.float16)  # 转换为 FP16
            outlier_indices = torch.nonzero(mask, as_tuple=False)
            
            # 将原位置置 0
            x[mask] = 0
        else:
            k = 0
            outlier_values = torch.empty(0)
            outlier_indices = torch.empty(0)
            mask = torch.zeros_like(x, dtype=torch.bool)
        
        outlier_size_bytes = 0
        if k > 0:
            # (A) 值的大小：强制按 FP16 计算 (k * 2)
            outlier_values_bytes = k * 2 
            
            # (B) 索引的大小：
            # 原方法假设可以将多维索引展平为 1 个 int32 (4 bytes)
            coo_indices_bytes = k * 4    
            
            # (C) Bitmap 大小：假设用 1 bit 表示一个位置
            bitmap_bytes = (total_elements + 7) // 8
            
            # (D) 取最小值：如果离群值很多，用 Bitmap 更省；如果很少，用坐标列表更省
            outlier_indices_bytes = min(coo_indices_bytes, bitmap_bytes)
            
            outlier_size_bytes = outlier_values_bytes + outlier_indices_bytes
            
        outlier_info = {
            "outlier_values": outlier_values,
            "outlier_indices": outlier_indices
        }

        return x, outlier_info, outlier_size_bytes

    def remove_outliers_by_indices(self, tensor, outlier_indices):
        """
        根据给定的索引，将张量中对应位置的值置为 0 (剔除离群值)。
        
        Args:
            tensor (torch.Tensor): 输入张量 (任意维度)。
            outlier_indices (torch.Tensor): 离群值坐标索引，Shape 为 [N, Dims]。
                                        (通常由 extract_union_outliers 返回)
            
        Returns:
            modified_tensor (torch.Tensor): 指定位置被置 0 后的新张量。
        """
        # 1. 复制张量，避免修改原数据
        modified_tensor = tensor.clone()
        
        # 2. 边界检查：如果没有索引，直接返回
        if outlier_indices.numel() == 0:
            return modified_tensor
            
        # 3. 索引转换
        # PyTorch 的 index_put_ 需要索引格式为 tuple(dim0_indices, dim1_indices, ...)
        # 输入的 outlier_indices 是 [N, D] 形状，需要转置为 [D, N] 并转为 tuple
        indices_tuple = tuple(outlier_indices.t())
        
        # 4. 执行置零操作
        # 使用 index_put_ 原地修改 clone 后的张量
        # 传入 0 的 tensor 形式以确保类型匹配
        zero_val = torch.tensor(0, dtype=modified_tensor.dtype, device=modified_tensor.device)
        modified_tensor.index_put_(indices_tuple, zero_val)
        
        return modified_tensor
     
    def restore_outliers(self, modified_tensor, outlier_info):
        """
        根据索引将离群值还原到张量中。
        
        Args:
            modified_tensor (torch.Tensor): 离群值位置为 0 的张量。
            outlier_values (torch.Tensor): 离群值数值。
            outlier_indices (torch.Tensor): 离群值坐标索引。
            
        Returns:
            restored_tensor (torch.Tensor): 还原后的完整张量。
        """
        # 复制一份以避免修改输入
        restored_tensor = modified_tensor.clone()
        outlier_values = outlier_info["outlier_values"]
        outlier_indices = outlier_info["outlier_indices"]
        
        # 边界检查
        if outlier_values.numel() == 0 or outlier_indices.numel() == 0:
            return restored_tensor
            
        # PyTorch 的高级索引需要将 [N, D] 的 indices 转置为 Tuple([N], [N], ...)
        # 例如：indices 是 [[0,1], [2,3]] -> 需要变成 ([0, 2], [1, 3])
        indices_tuple = tuple(outlier_indices.t())
        
        # 使用 index_put_ 将值填回 (或者直接用索引赋值)
        # index_put_ 能够很好地处理原地操作
        restored_tensor.index_put_(indices_tuple, outlier_values)
        
        return restored_tensor

    def simulate_quant_int8(self, hidden_states, mixed_precision, use_group, add_bytes=0):
        quant_res, size_str = self.quantize_fp16_int8(hidden_states, 128, 0.01, mixed_precision, use_group, add_bytes)
        dequant_res = self.dequantize_int8_fp16(quant_res)
        return dequant_res, size_str
    
    def simulate_quant_int4(self, hidden_states, mixed_precision, use_group, add_bytes=0):
        quant_res, size_str = self.quantize_fp16_int4(hidden_states, 128, 0.01, mixed_precision, use_group, add_bytes)
        dequant_res = self.dequantize_int4_fp16(quant_res)
        return dequant_res, size_str
    
    def simulate_quant_int2(self, hidden_states, mixed_precision, use_group, add_bytes=0):
        quant_res, size_str = self.quantize_fp16_int2(
            hidden_states, 
            group_size=128, 
            outlier_ratio=0.01, 
            mixed_precision=mixed_precision, 
            use_grouping=use_group, 
            add_bytes=add_bytes
        )
        dequant_res = self.dequantize_int2_fp16(quant_res)
        return dequant_res, size_str

    # Per-token quantization (no mixed precision, no grouping)
    def simulate_quant_int8_per_token(self, hidden_states, add_bytes=0):
        quant_res, size_str = self.quantize_per_token_int8(hidden_states, add_bytes)
        dequant_res = self.dequantize_per_token_int8(quant_res)
        return dequant_res, size_str

    def simulate_quant_int2_per_token(self, hidden_states, add_bytes=0):
        quant_res, size_str = self.quantize_per_token_int2(hidden_states, add_bytes)
        dequant_res = self.dequantize_per_token_int2(quant_res)
        return dequant_res, size_str
    
    def simulate_quant_int4_per_token(self, hidden_states, add_bytes=0):
        quant_res, size_str = self.quantize_per_token_int4(hidden_states, add_bytes)
        dequant_res = self.dequantize_per_token_int4(quant_res)
        return dequant_res, size_str
    
    def _fit_channel_affine(self, X, Y):
        """
        实时计算通道级仿射变换参数: Y[:, c] = a[c] * X[:, c] + b[c]
        
        Args:
            X: Tensor (B, L, H) - Layer 1 的激活（被拟合的源）
            Y: Tensor (B, L, H) - Layer 2 的激活（拟合的目标）
        
        Returns:
            slope: Tensor (H,) - 每个通道的斜率
            intercept: Tensor (H,) - 每个通道的截距
        """
        # 展平 batch 和 sequence 维度: (B, L, H) -> (B*L, H)
        X_flat = X.reshape(-1, X.shape[-1]).float()
        Y_flat = Y.reshape(-1, Y.shape[-1]).float()
        
        # 计算每个通道的均值
        X_mean = X_flat.mean(dim=0)  # (H,)
        Y_mean = Y_flat.mean(dim=0)  # (H,)
        
        # 计算协方差和方差
        numerator = ((X_flat - X_mean) * (Y_flat - Y_mean)).sum(dim=0)  # cov(X, Y)
        denominator = ((X_flat - X_mean) ** 2).sum(dim=0) + 1e-6        # var(X)
        
        # 计算斜率和截距
        slope = numerator / denominator
        intercept = Y_mean - slope * X_mean
        
        slope_bytes = slope.numel() * slope.element_size()
        intercept_bytes = intercept.numel() * intercept.element_size()
        
        total_bytes = slope_bytes + intercept_bytes
            
        return slope.half(), intercept.half(), total_bytes  # 转回 FP16 节省显存
    
    def _apply_channel_affine(self, X, slope, intercept):
        """
        应用通道级仿射变换
        
        Args:
            X: Tensor (B, L, H) - 输入激活
            slope: Tensor (H,) - 斜率
            intercept: Tensor (H,) - 截距
        
        Returns:
            X_transformed: Tensor (B, L, H) - 变换后的激活
        """
        return X * slope + intercept

    def affine_transform(self, hidden1, hidden2):
        slope, intercept, size_bytes = self._fit_channel_affine(hidden1, hidden2)
        hidden1_transformed = self._apply_channel_affine(hidden1, slope, intercept)
        return hidden1_transformed, size_bytes, slope, intercept

    # -----------------------------
    # Per-token INT8/INT4/INT2 APIs
    # -----------------------------
    def quantize_per_token_int8(self, tensor, add_bytes=0):
        x = tensor.clone().float()
        original_shape = x.shape
        total_elements = x.numel()

        # Flatten to [seq_len, hidden]
        x_flat = x.reshape(-1, original_shape[-1])
        seq_len, hidden = x_flat.shape

        # One group per token (no grouping)
        effective_group_size = hidden
        pad_len = 0

        # Shape to [seq_len, 1, hidden] for dequant symmetry
        x_groups = x_flat.view(seq_len, 1, hidden)

        # Per-token min/max
        min_val = x_groups.min(dim=-1, keepdim=True)[0]
        max_val = x_groups.max(dim=-1, keepdim=True)[0]
        range_val = (max_val - min_val).clamp(min=1e-6)

        base_bits = 8
        max_q_val = 2**base_bits - 1
        scale = range_val / max_q_val

        # Quantize
        x_q = ((x_groups - min_val) / scale).round().clamp(0, max_q_val).to(torch.uint8)

        # Size estimation
        body_payload_bytes = total_elements  # 1 byte per element
        body_metadata_bytes = min_val.numel() * 4  # store min/scale in FP16? dequant expects float; align with 4 bytes per value
        total_bytes = body_payload_bytes + body_metadata_bytes + add_bytes

        if total_bytes < 1024:
            size_str = f"{total_bytes:.0f} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes/1024:.2f} KB"
        else:
            size_str = f"{total_bytes/(1024**2):.2f} MB"

        quantized_data = {
            "q_data": x_q,
            "scales": scale.half(),
            "min_vals": min_val.half(),
            "outlier_indices": torch.empty(0, dtype=torch.long),
            "outlier_values": torch.empty(0, dtype=torch.float16),
            "original_shape": original_shape,
            "pad_len": pad_len,
            "mixed_precision": False,
            "effective_group_size": effective_group_size,
        }

        return quantized_data, size_str

    def dequantize_per_token_int8(self, quantized_data):
        x_q = quantized_data["q_data"]
        scale = quantized_data["scales"].float()
        min_val = quantized_data["min_vals"].float()
        original_shape = quantized_data["original_shape"]
        pad_len = quantized_data["pad_len"]

        x_recon_groups = x_q.float() * scale + min_val
        seq_len = x_recon_groups.shape[0]
        padded_hidden = x_recon_groups.shape[1] * x_recon_groups.shape[2]
        x_recon_flat = x_recon_groups.view(seq_len, padded_hidden)

        if pad_len > 0:
            actual_hidden = padded_hidden - pad_len
            x_recon_flat = x_recon_flat[:, :actual_hidden]

        x_recon = x_recon_flat.reshape(original_shape)
        return x_recon.to(torch.float16)

    def quantize_per_token_int2(self, tensor, add_bytes=0):
        x = tensor.clone().float()
        original_shape = x.shape
        total_elements = x.numel()

        # Flatten
        x_flat = x.reshape(-1, original_shape[-1])
        seq_len, hidden = x_flat.shape

        # Ensure group size multiple of 4 for packing
        effective_group_size = hidden
        if effective_group_size % 4 != 0:
            effective_group_size += (4 - effective_group_size % 4)

        pad_len = (effective_group_size - hidden % effective_group_size) % effective_group_size
        if pad_len > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))

        num_groups = x_flat.shape[1] // effective_group_size
        x_groups = x_flat.view(seq_len, num_groups, effective_group_size)

        # Per-token min/max (one group per token)
        min_val = x_groups.min(dim=-1, keepdim=True)[0]
        max_val = x_groups.max(dim=-1, keepdim=True)[0]
        range_val = (max_val - min_val).clamp(min=1e-6)
        max_q_val = 3
        scale = range_val / max_q_val

        x_q = ((x_groups - min_val) / scale).round().clamp(0, max_q_val).to(torch.uint8)

        # Pack 4x2-bit into uint8
        v0 = x_q[..., 0::4] << 6
        v1 = x_q[..., 1::4] << 4
        v2 = x_q[..., 2::4] << 2
        v3 = x_q[..., 3::4]
        x_packed = v0 | v1 | v2 | v3

        body_payload_bytes = total_elements * 0.25
        body_metadata_bytes = min_val.numel() * 4
        total_bytes = body_payload_bytes + body_metadata_bytes + add_bytes

        size_str = (
            f"{total_bytes:.0f} B" if total_bytes < 1024 else (
                f"{total_bytes/1024:.2f} KB" if total_bytes < 1024**2 else f"{total_bytes/(1024**2):.2f} MB"
            )
        )

        quantized_data = {
            "q_data": x_packed,
            "scales": scale.half(),
            "min_vals": min_val.half(),
            "outlier_indices": torch.empty(0, dtype=torch.long),
            "outlier_values": torch.empty(0, dtype=torch.float16),
            "original_shape": original_shape,
            "pad_len": pad_len,
            "mixed_precision": False,
            "effective_group_size": effective_group_size,
        }

        return quantized_data, size_str

    def dequantize_per_token_int2(self, quantized_data):
        x_packed = quantized_data["q_data"]
        scale = quantized_data["scales"].float()
        min_val = quantized_data["min_vals"].float()
        original_shape = quantized_data["original_shape"]
        pad_len = quantized_data["pad_len"]

        v0 = (x_packed >> 6) & 0x03
        v1 = (x_packed >> 4) & 0x03
        v2 = (x_packed >> 2) & 0x03
        v3 = x_packed & 0x03
        x_q = torch.stack((v0, v1, v2, v3), dim=-1).flatten(start_dim=-2)

        x_recon_groups = x_q.float() * scale + min_val
        seq_len = x_recon_groups.shape[0]
        padded_hidden = x_recon_groups.shape[1] * x_recon_groups.shape[2]
        x_recon_flat = x_recon_groups.view(seq_len, padded_hidden)

        if pad_len > 0:
            actual_hidden = padded_hidden - pad_len
            x_recon_flat = x_recon_flat[:, :actual_hidden]

        x_recon = x_recon_flat.reshape(original_shape)
        return x_recon.to(torch.float16)

    def quantize_per_token_int4(self, tensor, add_bytes=0):
        x = tensor.clone().float()
        original_shape = x.shape
        total_elements = x.numel()

        # Flatten
        x_flat = x.reshape(-1, original_shape[-1])
        seq_len, hidden = x_flat.shape

        # Ensure even group size for 4-bit packing
        effective_group_size = hidden
        if effective_group_size % 2 != 0:
            effective_group_size += 1

        pad_len = (effective_group_size - hidden % effective_group_size) % effective_group_size
        if pad_len > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))

        num_groups = x_flat.shape[1] // effective_group_size
        x_groups = x_flat.view(seq_len, num_groups, effective_group_size)

        # Per-token min/max
        min_val = x_groups.min(dim=-1, keepdim=True)[0]
        max_val = x_groups.max(dim=-1, keepdim=True)[0]
        range_val = (max_val - min_val).clamp(min=1e-6)
        max_q_val = 15  # 2^4 - 1
        scale = range_val / max_q_val

        x_q = ((x_groups - min_val) / scale).round().clamp(0, max_q_val).to(torch.uint8)

        # Pack 2 nibbles into a byte
        x_q_high = x_q[..., 0::2]
        x_q_low = x_q[..., 1::2]
        x_packed = (x_q_high << 4) | x_q_low

        body_payload_bytes = total_elements * 0.5
        body_metadata_bytes = min_val.numel() * 4
        total_bytes = body_payload_bytes + body_metadata_bytes + add_bytes

        if total_bytes < 1024:
            size_str = f"{total_bytes:.0f} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes/1024:.2f} KB"
        else:
            size_str = f"{total_bytes/(1024**2):.2f} MB"

        quantized_data = {
            "q_data": x_packed,
            "scales": scale.half(),
            "min_vals": min_val.half(),
            "outlier_indices": torch.empty(0, dtype=torch.long),
            "outlier_values": torch.empty(0, dtype=torch.float16),
            "original_shape": original_shape,
            "pad_len": pad_len,
            "mixed_precision": False,
            "effective_group_size": effective_group_size,
        }

        return quantized_data, size_str

    def dequantize_per_token_int4(self, quantized_data):
        x_packed = quantized_data["q_data"]
        scale = quantized_data["scales"].float()
        min_val = quantized_data["min_vals"].float()
        original_shape = quantized_data["original_shape"]
        pad_len = quantized_data["pad_len"]

        # Unpack high/low nibbles
        x_q_high = (x_packed >> 4) & 0x0F
        x_q_low = x_packed & 0x0F
        x_q_stacked = torch.stack((x_q_high, x_q_low), dim=-1)
        x_q = x_q_stacked.flatten(start_dim=-2)

        x_recon_groups = x_q.float() * scale + min_val
        seq_len = x_recon_groups.shape[0]
        padded_hidden = x_recon_groups.shape[1] * x_recon_groups.shape[2]
        x_recon_flat = x_recon_groups.view(seq_len, padded_hidden)

        if pad_len > 0:
            actual_hidden = padded_hidden - pad_len
            x_recon_flat = x_recon_flat[:, :actual_hidden]

        x_recon = x_recon_flat.reshape(original_shape)
        return x_recon.to(torch.float16)

    # ==========================================
    # FP8 Per-Token 量化 (使用原生 float8_e4m3fn)
    # ==========================================
    def simulate_quant_fp8_per_token(self, hidden_states, add_bytes=0):
        quant_res, size_str = self.quantize_per_token_fp8(hidden_states, add_bytes)
        dequant_res = self.dequantize_per_token_fp8(quant_res)
        return dequant_res, size_str

    def quantize_per_token_fp8(self, tensor, add_bytes=0):
        x = tensor.clone().float()
        original_shape = x.shape
        total_elements = x.numel()

        # Flatten 
        x_flat = x.reshape(-1, original_shape[-1])
        
        # FP8 E4M3 的最大表示范围大约是 448
        # Per-Token Scaling: 
        # 1. 找到每行的绝对值最大值
        # 2. 计算 scale，将该行映射到 FP8 的最佳动态范围 [-448, 448]
        max_val = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        fp8_max = 448.0 
        scale = max_val / fp8_max

        # 量化: 除以 scale -> 转换到 float8
        # 注意：需要 PyTorch 支持 float8 (torch >= 2.1 且 GPU 支持)
        # 如果不支持，这里会报错，需要回退到 int8 模拟
        try:
            x_scaled = (x_flat / scale)
            x_q = x_scaled.to(torch.float8_e4m3fn) 
        except (AttributeError, TypeError):
            # Fallback for CPU or older torch: simulate with int8 but labeling as fp8 logic
            print("Warning: torch.float8_e4m3fn not supported/available, simulating with int8 container.")
            x_q = x_scaled.clamp(-fp8_max, fp8_max).round().to(torch.int8)

        # 统计体积
        body_payload_bytes = total_elements * 1  # FP8 is 1 byte per element
        body_metadata_bytes = scale.numel() * 2  # FP16 scale
        total_bytes = body_payload_bytes + body_metadata_bytes + add_bytes

        if total_bytes < 1024:
            size_str = f"{total_bytes:.0f} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes/1024:.2f} KB"
        else:
            size_str = f"{total_bytes/(1024**2):.2f} MB"

        quantized_data = {
            "q_data": x_q,
            "scales": scale.half(),
            "original_shape": original_shape,
        }
        return quantized_data, size_str

    def dequantize_per_token_fp8(self, quantized_data):
        x_q = quantized_data["q_data"]
        scale = quantized_data["scales"].float()
        original_shape = quantized_data["original_shape"]

        # Dequantize: Cast back to float and multiply by scale
        x_recon_flat = x_q.float() * scale
        x_recon = x_recon_flat.reshape(original_shape)
        
        return x_recon.to(torch.float16)

    # ==========================================
    # FP4 Per-Token 量化 (模拟 E2M1, Bit-Packed)
    # ==========================================
    def simulate_quant_fp4_per_token(self, hidden_states, add_bytes=0):
        quant_res, size_str = self.quantize_per_token_fp4(hidden_states, add_bytes)
        dequant_res = self.dequantize_per_token_fp4(quant_res)
        return dequant_res, size_str

    def quantize_per_token_fp4(self, tensor, add_bytes=0):
        x = tensor.clone().float()
        original_shape = x.shape
        total_elements = x.numel()

        # =====================================================
        # 【必须复制的核心逻辑】定义 FP4 E2M1 标准码本
        # =====================================================
        # 这些是 FP4 E2M1 标准对应的 16 个浮点数值
        # 包含了负数部分和正数部分
        fp4_values = [
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,  # 索引 0-7
            0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0, 6.0   # 索引 8-15
        ]
        # 【关键】创建 tensor 时，必须指定 device=x.device，否则会跨设备报错
        codebook = torch.tensor(fp4_values, device=x.device, dtype=torch.float32)
        # =====================================================

        # Flatten
        x_flat = x.reshape(-1, original_shape[-1])
        seq_len, hidden = x_flat.shape

        # 补齐 (Padding) 以便两两打包
        effective_group_size = hidden
        if effective_group_size % 2 != 0:
            effective_group_size += 1
        pad_len = effective_group_size - hidden
        if pad_len > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))

        # Scale 计算 (Max-abs)
        fp4_max_val = 6.0
        abs_max = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
        scale = abs_max / fp4_max_val
        
        # 归一化
        x_scaled = x_flat / scale

        # 查表映射 (Mapping)
        # 寻找 x_scaled 与 codebook 中哪个数最接近
        # diff shape: (rows, cols, 16)
        x_expanded = x_scaled.unsqueeze(-1)
        diff = (x_expanded - codebook).abs()
        x_indices = torch.argmin(diff, dim=-1).to(torch.uint8) # 得到 0-15 的索引

        # 打包 (Packing): 2个4-bit拼成1个8-bit
        x_q_high = x_indices[..., 0::2]
        x_q_low = x_indices[..., 1::2]
        x_packed = (x_q_high << 4) | x_q_low

        # 计算体积字符串
        body_payload_bytes = total_elements * 0.5
        body_metadata_bytes = scale.numel() * 2 
        total_bytes = body_payload_bytes + body_metadata_bytes + add_bytes

        if total_bytes < 1024:
            size_str = f"{total_bytes:.0f} B"
        elif total_bytes < 1024**2:
            size_str = f"{total_bytes/1024:.2f} KB"
        else:
            size_str = f"{total_bytes/(1024**2):.2f} MB"

        quantized_data = {
            "q_data": x_packed,
            "scales": scale.half(),
            "original_shape": original_shape,
            "pad_len": pad_len,
            # 把 codebook 存入结果中，这样反量化时直接从这里取，不需要反量化函数再定义一遍
            "codebook": codebook 
        }

        return quantized_data, size_str

    def dequantize_per_token_fp4(self, quantized_data):
        x_packed = quantized_data["q_data"]
        scale = quantized_data["scales"].float()
        original_shape = quantized_data["original_shape"]
        pad_len = quantized_data["pad_len"]
        
        # 直接从输入字典里拿 codebook，不需要在函数里重新定义
        codebook = quantized_data["codebook"]

        # 解包 (Unpacking)
        x_index_high = (x_packed >> 4) & 0x0F
        x_index_low = x_packed & 0x0F
        x_indices = torch.stack((x_index_high, x_index_low), dim=-1).flatten(start_dim=-2)

        # 查表恢复 (Lookup)
        x_recon_flat = codebook[x_indices.long()]
        
        # 移除 Padding
        if pad_len > 0:
            x_recon_flat = x_recon_flat[:, :-pad_len]
            
        # 反归一化
        x_recon = x_recon_flat * scale
        
        return x_recon.reshape(original_shape).to(torch.float16)