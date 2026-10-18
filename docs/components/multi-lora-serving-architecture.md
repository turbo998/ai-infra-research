# 千级多租户 Serverless Multi-LoRA 高密服务架构：`S-LoRA` 统一分页池、`vLLM Punica` `SGMV` 算子与 Zipfian 异步预取

- **关联问题 ID**：`multi-lora-serverless-adapter-hotswap-punica-slora` (`priority: 17`)
- **关联实验**：[`experiments/multi_lora_serving/run.py`](../../experiments/multi_lora_serving/run.py) · [`results.json`](../../experiments/multi_lora_serving/results.json)
- **一手验证源码**：
  - `vllm-project/vllm` (`commit 82a24bb82dd9344cce7a43195bb51a2390adf00a`)：
    - `vllm/lora/punica_wrapper/punica_gpu.py` (`blob 8604e747f9dfc4b28b075c06f62dfbb5677bfcb4`)
    - `vllm/lora/ops/triton_ops/lora_shrink_op.py` (`blob abe63bd41bbfc009c8f96983e33aad4048d9808e`)
    - `vllm/lora/ops/triton_ops/lora_expand_op.py` (`blob 05fea52a71eb2aab259b22f045cf995c1fae6659`)
    - `vllm/lora/ops/triton_ops/lora_kernel_metadata.py` (`blob e4bd7bfacdf2a8a95e62a1d9d3d38ac2390adf00a` / `e4bd7bfacdf2a8a95e62a1d9d3d38ac23975dbca`)
    - `vllm/lora/model_manager.py` (`blob a7f9f02e77af8d7bcdb63d7ba965546b02aab50e`)

---

## 1. 架构背景：为何在单集群服务 1,000+ 微调模型时不能合并权重（Merged Weights）？

在企业级 SaaS 与 Agent 平台中，数百至上千个租户基于同一基座模型（如 `Llama-3-70B` 或 `Qwen2.5-72B`）挂载各自专属的动态秩 LoRA 适配器（`rank r ∈ {8, 16, 64, 128}`）。若将 LoRA 权重合并进基座模型（$W' = W_0 + s \cdot B A$）：
1. **显存爆炸**：每个合并后的 `70B` BF16 副本需独占 `140 GB` HBM，单台 `8x H100 (640 GB)` 仅能常驻 `4` 个租户模型；
2. **Batch 碎片化（Unfused Batching Collapse）**：即便保持基座权重只存一份，若在推理时按租户拆分 Batch 分别调用基座 GEMM（$Y_i = X_i W_0 + s X_i A_i B_i$），当并发活跃适配器数达到 `64` 时，每步 Decode 需要重复从 HBM 读取 `64` 次 `140 GB` 基座权重，导致单步耗时从 `12.6 ms` 飙升至 `615.7 ms`，吞吐暴跌 **`36.5x`**！

---

## 2. 一手源码解剖：`vLLM Punica` Triton `SGMV` 算子与 `LoRAKernelMeta` 元数据管线

### 2.1 基座共享 GEMM + 分段聚集矩阵乘（`SGMV` Shrink / Expand）

Punica 与 S-LoRA 利用分配律将所有租户 Token 合并为一个大 Batch $X \in \mathbb{R}^{B \times H}$ 共享单次高算术强度基座 GEMM，随后仅对低秩分支叠加 `SGMV`（Segmented Gather Matrix-Vector Multiplication）：

$$Y = \underbrace{X W_0}_{\text{All tokens share 1 Base GEMM}} + \sum_{k \in \text{ActiveLoRAs}} s_k \cdot \big(\underbrace{X_{\mathcal{I}_k} A_k}_{\text{lora\_shrink } (H \to r_k)}\big) \underbrace{B_k}_{\text{lora\_expand } (r_k \to H_{\text{out}})}$$

在 `vllm/lora/punica_wrapper/punica_gpu.py`（第 `91–173` 行）与 `vllm/lora/ops/triton_ops/` 中，这一计算由三阶段流水线完成：

1. **`LoRAKernelMeta.prepare_tensors` 零 Python 循环元数据重排**（`lora_kernel_metadata.py` 第 `107–166` 行）：
   - 检查 `no_lora = torch.all(token_lora_mapping == -1)` 并写入 CPU 标量张量 `no_lora_flag_cpu[0]`（使 `torch.compile` 图捕获无需重编译即可在算子内部快速跳过无 LoRA 批次）；
   - 执行 `torch.sort(token_lora_mapping, stable=True)` 生成按适配器聚类的 Token 索引 `token_indices_sorted_by_lora_ids`；
   - 执行 `torch.unique(token_lora_mapping, sorted=True, return_counts=True)` 与 `torch.cumsum` 生成 `active_lora_ids`、`num_tokens_per_lora` 与 `lora_token_start_loc`；
   - 当启用 `specialize_active_lora` 时，通过 `bisect.bisect_left(self.captured_lora_counts, num_active_loras)` 将活跃 LoRA 数量向上对齐至 CUDA Graph 捕获桶，避免 Grid 维度变化触发 CUDA Graph 失效。
2. **`_lora_shrink_kernel` (`H -> r` 降维，大规约维 `SPLIT_K`)**（`lora_shrink_op.py` 第 `26–190` 行）：
   - 输入特征维度 $K = H$（如 `4096` 或 `8192`）远大于输出秩 $N = r \in [8, 64]$。为填满 SM 并避免原子加引起的浮点非确定性，vLLM 支持在 $K$ 维按 `SPLIT_K`（Batch-invariant 模式固定 `_BI_SPLIT_K = 8`）切分 Partial Sum，并由 `_lora_shrink_reduce_kernel` 确定性归约乘上 `scaling`。
   - 线程块在 3D Grid `(pid_sk_m_n, slice_id, lora_idx)` 上启动：通过 `lora_token_start_loc[lora_idx]` 定位属于当前适配器的 Token 行索引数组 `ram`，通过间接寻址 Gather 对应行与 `lora_a_stacked[slice_id][lora_id]` 相乘。
3. **`_lora_expand_kernel` (`r -> H_out` 升维，支持 GQA 异构 Slice 步幅)**（`lora_expand_op.py` 第 `24–133` 行）：
   - 在融合 `QKV` 投影（GQA 下 $Q$ 与 $K, V$ 输出维度不同）或 `Gate/Up` 投影中，`SAME_STRIDE` 为 `False` 时内核动态读取 `curr_N = tl.load(output_hs_ptr + slice_id)` 与偏移 `slice_start_loc`，直接通过 `ram` Scatter-Add 回主输出张量 `Y[:, offset:offset+curr_N]`；
   - 支持 `add_inputs=False` 双流模式（`punica_gpu.py` 第 `148–150` 行注释：当 Base GEMM 与 LoRA SGMV 运行在不同 CUDA Stream 时，分别写出再归约）。

---

## 3. `S-LoRA` 统一分页池（Unified Paging）与 Zipfian 异步预取

### 3.1 静态最大秩 Slab 的碎片危机 vs `S-LoRA` 统一分页

当集群同时服务 $r \in \{8, 16, 64, 128\}$ 的异构租户（例如 50% $r=8$、30% $r=16$、15% $r=64$、5% $r=128$，加权平均秩 $\bar{r} = 24.8$）时：
- **静态 Max-Rank Slab（固定 `max_lora_rank=128` 预留槽位 + 独立 KV 池）**：每个 GPU Slot 必须按 $r_{\max}=128$ 静态分配连续张量，导致低秩适配器槽位内部浪费高达 `80.6%`，叠加 KV 与 LoRA 静态分区边界，总体有效 HBM 损失达 **`31.4%`**。
- **`S-LoRA` Unified Paging Pool**：由于 KV Cache Page（形状为 `[Page_Size, Num_KV_Heads, Head_Dim]`，展平后宽度为 $H$）与 LoRA 权重矩阵 $A \in \mathbb{R}^{r \times H}, B \in \mathbb{R}^{H \times r}$ **共享相同的隐藏层步幅维度 $H$**，S-LoRA 将剩余 HBM 统一划分为步幅为 $H$ 的物理 Page，由统一页表同时按需分配 KV Pages（$P$ 行）与 LoRA Adapter Pages（$r_i$ 行）。外部碎片率从 `31.4%` 骤降至 **`1.6%`**，可并发承载的 KV Cache Token 容量从 `51,712` 提升至 **`131,712`（`+154.7%`）**！

### 3.2 `AdapterLRUCache` 与 Zipfian (`s=1.2`) 异步预取

在 `vllm/lora/model_manager.py`（第 `62–71`、`111–118` 行）中，`LoRAModelManager` 维护两级 `AdapterLRUCache`：CPU Pinned Memory 缓存池（`_registered_adapters`）与 GPU HBM 活跃槽位池（`_active_adapters`）。
- 在 `1,000` 个注册适配器服从 `Zipf(s=1.2)` 流量分布、`128` 个 GPU 活跃槽位下，纯被动 LRU 命中率为 `89.73%`，但 `10.27%` 的冷命中会触发同步 PCIe Gen5 x16 权重拷贝（`~6.8 ms`），导致 P99 TTFT 劣化至 `54.32 ms`。
- 结合调度队列前瞻（Queue-Lookahead）与 Copy Engine 双流异步预取后，当请求还在等待当前 Prefill 批次完成时，下一个批次所需的冷适配器已通过独立 DMA 流提前换入备用 Slot，有效命中率升至 **`95.64%`**，P99 TTFT 降至 **`39.22 ms`（降低 `27.8%`）**。

---

## 4. 离线实验验证结果（`experiments/multi_lora_serving`）

### 4.1 并发活跃适配器扩展性（`Batch=256 tokens`, `r=16`, `Llama-3-70B TP=4`）

| 并发活跃适配器数 | 按适配器拆批 (`Unfused`) 延迟 | `Punica/S-LoRA` (`Base + SGMV`) 延迟 | `Unfused` 吞吐 (`tok/s`) | `Punica/S-LoRA` 吞吐 (`tok/s`) | 吞吐加速比 | `SGMV` 占总耗时比 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `9.62 ms` | `13.85 ms` | `26,600.2` | `18,483.8` | `0.69x` | `14.8%` |
| `4` | `38.50 ms` | `13.99 ms` | `6,650.0` | `18,293.5` | **`2.75x`** | `15.7%` |
| `16` | `153.98 ms` | `14.57 ms` | `1,662.5` | `17,570.3` | **`10.57x`** | `19.0%` |
| `32` | `307.97 ms` | `15.34 ms` | `831.3` | `16,690.6` | **`20.08x`** | `23.1%` |
| `64` | `615.94 ms` | `16.87 ms` | `415.6` | **`15,171.3`** | **`36.50x`** | `30.1%` |
| `128` | `1,231.87 ms` | `19.95 ms` | `207.8` | **`12,834.6`** | **`61.76x`** | `40.8%` |

### 4.2 LoRA 秩盈亏平衡拐点 $r^*$（`Batch=256 tokens`）

| LoRA 秩 $r$ | 单适配器体积 (`BF16`) | `SGMV` 相对基座 GEMM 额外开销 | `Punica/S-LoRA` 吞吐 (`tok/s`) | 合并权重多实例集群吞吐 (`tok/s`) | 架构选型建议 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `8` | `41.6 MiB` | `+8.9%` | `20,031.3` | `16,250.9` | `Punica/S-LoRA Shared Base + SGMV` |
| `16` | `83.2 MiB` | `+12.3%` | `19,320.8` | `16,250.9` | `Punica/S-LoRA Shared Base + SGMV` |
| `32` | `166.4 MiB` | `+19.1%` | `18,040.9` | `16,250.9` | `Punica/S-LoRA Shared Base + SGMV` |
| **`64` ($r^*$)** | **`332.8 MiB`** | **`+32.6%`** | **`16,357.8`** | **`16,250.9`** | **`Punica/S-LoRA Shared Base + SGMV`（盈亏平衡拐点 $r^* = 64$）** |
| `96` | `499.2 MiB` | `+46.2%` | `14,962.0` | `16,250.9` | `Merged-Weight Dedicated Pool (Rank > r*=64)` |
| `128` | `665.6 MiB` | `+59.7%` | `13,785.7` | `16,250.9` | `Merged-Weight Dedicated Pool (Rank > r*=64)` |

---

## 5. 解决方案架构师行动指南与设计练习

1. **按秩分流（Rank-Tiered Routing）准则**：
   - 对于 $r \le 64$ 的长尾微调租户，统一收敛至 **共享基座 + `S-LoRA` 统一分页 + `Punica SGMV`** 集群；
   - 对于 $r > 64$（如 $r = 128, 256$）且 QPS 稳定的头部租户，`SGMV` 的非连续 Gather/Scatter 与双算子开销超过 `+32.6%`，应将其权重静态 Merge 进基座并路由至专属实例池。
2. **可证伪后续问题**：在 MoE 模型（如 DeepSeek-V3 `256` 专家）上挂载专家级 LoRA (`FusedMoE3DWithLoRA`) 时，`EP All-to-All` 与 `ActiveLoRAs x ActiveExperts` 的笛卡尔积稀疏性会使盈亏平衡拐点 $r^*$ 下降到多少？
