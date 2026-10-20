# TorchTitan Kimi K3 MXFP4 权重适配器 (`KimiK3StateDictAdapter`)、DeepSeek-V3 HF 导出索引对齐与 GraphTrainer `SimpleFSDP` Meta 设备零物化检查机制

> **English Version**: [torchtitan-kimi-k3-mxfp4-adapter-graph-fsdp-meta-inspect.en.md](torchtitan-kimi-k3-mxfp4-adapter-graph-fsdp-meta-inspect.en.md) | **关联实验**: [`experiments/torchtitan_kimi_k3_mxfp4_adapter_fsdp/README.md`](../../experiments/torchtitan_kimi_k3_mxfp4_adapter_fsdp/README.md)

### 一句话结论

PyTorch TorchTitan 在混合注意力与超大规模 MoE 训练/检查点栈中合入了三项关键工程契约：一是 **`KimiK3StateDictAdapter`（`#5337`）** 通过 `QuantizedHuggingFaceStorageReader(target_dtype=torch.bfloat16)` 在线将 `compressed-tensors` MXFP4 路由专家（`.weight_packed` / `.weight_scale`）反量化为 BF16，并对 KDA（`KimiDeltaAttention`）层的 `A_log`（从 `[n_heads]` 零填充至 `[head_dim]`）与 `dt_bias`（从 `[n_heads, head_dim]` 展平为 1D）执行先重分布为 `Replicate()`、加载后再恢复原始 `DTensor` Mesh/Placement 的无损往返；二是 **`DeepSeekV3StateDictAdapter`（`#5065`）** 从官方 `91,991` 项 `model.safetensors.index.json` 中精确剔除 `45,808` 项未保存的 FP8 `.weight_scale_inv` 与 `788` 项未启用的 MTP 层（精确对齐 `45,395` / `46,183` 个写出张量，彻底修复 DCP 合并生成 `{"shape": [], "dtype": ""}` 导致 `safetensors` 报 `unknown variant ''` 崩溃），同时为 MTP 层克隆独立的 `embed_tokens`/`lm_head` 存储副本（671B BF16 约 `+3.71 GB`）；三是 **`GraphTrainer` `SimpleFSDP`（`#5560`）** 在 `linear_param_shard_placements(model)` 扫描时包裹 `with disable_active_parametrization():`，阻止 `GraphPP + EP` 复用模块在 `meta` 设备阶段提前触发 `ReplicateComputation` 与 MXFP8 `_build_operands()`。

### 组件责任与上游主线位置

| 模块与文件 | 上游仓库与 Commit / Blob SHA | 核心工程职责 |
| :--- | :--- | :--- |
| `torchtitan/models/kimi_k3/state_dict_adapter.py` | `pytorch/torchtitan` commit `7d563223d4ee10c0b8a7fc70d0cd119bcbcff93f` / blob `221fab1377ea753cb199f4d3de2f85ae91889014` | 实现 Kimi K3 语言+视觉多模态与 MLA/KDA 混合层的 HF 双向状态字典适配、MXFP4 `weight_packed` 加载规划、KDA `A_log`/`dt_bias` DTensor 布局保存与恢复、以及 Layer-0 闲置 AttnRes 占位符注入。 |
| `torchtitan/models/deepseek_v3/state_dict_adapter.py` & `deepseek_v4/state_dict_adapter.py` | `pytorch/torchtitan` commit `1d581795cd014a46e871cc9fc605325fbbceee85` / blob `ac991872452b315d263ca36bc790cf9557557a84` | 过滤 `fqn_to_index_mapping` 中的 `.weight_scale_inv` 与超界 MTP 层号，并在 `to_hf()` 中通过 `_MTP_SHARED_COPIES` 克隆独立存储的 MTP Embedding/Head 副本。 |
| `torchtitan/experiments/graph_trainer/simple_fsdp.py` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `3327d72b0708477e75e75c62241cb63b8fdffd18` | 在 `data_parallel()` 推断线性层参数分片放置策略时启用 `disable_active_parametrization()` 上下文，防止 `meta` 设备上的量化/EP 参数在检查期间物化计算操作数。 |

### 核心实现路径、数据布局与同步契约

1. **Kimi K3 MXFP4 在线反量化与 KDA `DTensor` 坐标变换（`kimi_k3/state_dict_adapter.py:118-395`）**：
   - **MXFP4 索引重映射**：发布的 Kimi K3 检查点将路由专家存储为 `compressed-tensors` MXFP4 格式（键名为 `*.weight_packed` 并附带 `*.weight_scale`）。由于 TorchTitan 通过 `QuantizedHuggingFaceStorageReader(path, thread_count=4, target_dtype=torch.bfloat16)` 在加载时将其反量化为 BF16，且保存 HF 检查点时写出未量化 BF16 权重，构造函数将 `fqn_to_index_mapping` 中的 `_packed` 后缀剥离、丢弃 `.weight_scale`，并在纯文本配置（`vision_encoder is None`）下过滤 `vision_tower.*` 与 `mm_projector.*`。
   - **KDA `A_log` 填充与 `dt_bias` 展平的 `DTensor` 保护**：Kimi K3 交错使用 MLA 与 KDA（Kimi Delta Attention）层。在发布的 MXFP4 检查点中，KDA 的 `A_log` 从 `[n_heads]` 被零填充到了 `[head_dim]`，且 `dt_bias` 从 `[n_heads, head_dim]` 被展平为 1D `[n_heads * head_dim]`。因为 `pad` 和 `reshape(-1)` 会改变张量坐标并破坏分片 `DTensor` 语义，`to_hf()` 先将原始 `(device_mesh, placements)` 记录到 `self._kda_parameter_layouts[key]`，将张量重分布为全网格 `(Replicate(),) * ndim` 后再执行填充/展平供 DCP 生成加载计划；随后在 `from_hf()` 中先切片 `value[:n_heads]` 或重塑为 `[n_heads, head_dim]`，再调用 `value.redistribute(device_mesh=layout[0], placements=layout[1])` 恢复 TP 分片。
   - **Layer-0 AttnRes 占位符**：HF 检查点包含第 0 层未使用的 `self_attention_res_norm.weight` 与 `self_attention_res_proj.weight`。`to_hf()` 仅在拥有第 0 层的 PP Stage 上利用 `layers.0.ffn_res_norm/proj.weight` 的形状生成 `ones_like` / `zeros_like` 占位符，`from_hf()` 则通过 `_UNUSED_HF_LAYER_ZERO_ATTN_RES_KEYS` 直接忽略。
2. **DeepSeek-V3 `91,991` 项 HF 导出索引过滤与 MTP 共享权重克隆（`deepseek_v3/state_dict_adapter.py:29-65, 248-290`）**：
   - **为什么未过滤的 `fqn_to_index_mapping` 会导致 `safetensors` 损坏**：`HuggingFaceStorageWriter` 在合并（Consolidation）阶段会为 `fqn_to_index_mapping` 中的每一个键预先在目标分片文件头写入一条元数据；如果某个键在 `to_hf()` 中从未写出，合并器会留下空条目 `{"shape": [], "dtype": ""}`，导致 `safetensors` 重新打开该文件时报 `SafetensorError: unknown variant ''`。
   - 官方 `deepseek-ai/DeepSeek-V3` 的 `model.safetensors.index.json` 共包含 `91,991` 个条目，其中 `45,808` 个为 FP8 `.weight_scale_inv`，`788` 个为第 61 层 MTP（`model.layers.61.*`）。过滤掉 `.weight_scale_inv` 以及 `layer_id >= len(layers) + len(mtp_layers)` 的条目后，映射表与 `to_hf()` 实际写出的键集合实现**严格双射**：`num_mtp_layers = 0` 时精确为 `45,395` 项，`num_mtp_layers = 1` 时精确为 `46,183` 项。
   - **MTP 共享权重独立存储克隆（`_MTP_SHARED_COPIES`）**：当启用 MTP（`num_mtp_layers >= 1`）时，HF 格式在每个 MTP 层下保存了主嵌入层与输出头的副本（`model.layers.61.embed_tokens.weight` 与 `model.layers.61.shared_head.head.weight`），而 TorchTitan 内部共享同一个 `tok_embeddings.weight` 和 `lm_head.weight`。由于 `safetensors` 拒绝在同一文件中序列化共享底层 Storage 的别名张量，`to_hf()` 显式调用 `state_dict[tt_key].clone()` 生成独立存储副本（对 671B BF16 模型，$2 \times 129,280 \times 7,168 \times 2 = 3,707,232,256\text{ B} \approx 3.71\text{ GB}$），并在 `from_hf()` 中跳过这些副本。
3. **`GraphTrainer` `SimpleFSDP` 在 `meta` 设备上的零物化检查（`simple_fsdp.py:533-540`）**：
   - 在 `graph_trainer_dist_moe_mxfp8_fsdp_ep_pp`（B200 CI）中，模型首先在 `meta` 设备上构建（尚未调用 `to_empty(device="cuda")`）。当 `GraphPP` 或 `EP` 已经对部分子模块注册了 `SimpleFSDP` 的 `ReplicateComputation` 参数化钩子后，外层再次调用 `data_parallel(model)` 会默认执行 `linear_param_shard_placements(model)` 遍历并读取 `child.weight`。
   - 若不禁用参数化钩子，读取 `child.weight` 会立即触发 `ReplicateComputation.forward` 与 `_ShardedFSDPTensor._build_operands()`，进而调用 `dist_moe` 的 MXFP8 内核准备逻辑；而此时权重仍在 `meta` 设备上（`weight.is_cuda == False`），触发 `ValueError: weight must be contiguous CUDA storage`。通过 `with disable_active_parametrization():` 包裹 `linear_param_shard_placements(model)`，检查过程直接读取底层的原始分片参数，将计算操作数的物化严格推迟到真实的前向图追踪阶段。

### 关键公式、单位与工程假设

1. **DeepSeek-V3 HF 导出索引精确平衡方程**：
   $$N_{\text{saved}}(L_{\text{mtp}}) = N_{\text{index\_total}} - N_{\text{fp8\_scale\_inv}} - \mathbb{I}(L_{\text{mtp}} == 0) \cdot N_{\text{mtp\_layer61}}$$
   代入 $N_{\text{index\_total}} = 91,991$、$N_{\text{fp8\_scale\_inv}} = 45,808$、$N_{\text{mtp\_layer61}} = 788$，得 $N_{\text{saved}}(0) = 45,395$，$N_{\text{saved}}(1) = 46,183$。
2. **MTP 共享 Embedding/Head 独立克隆字节开销**：
   $$\text{Bytes}_{\text{MTP\_clone}} = 2 \cdot L_{\text{mtp}} \cdot V \cdot H \cdot \text{sizeof}(\text{dtype})$$
   当 $L_{\text{mtp}} = 1, V = 129,280, H = 7,168, \text{BF16} = 2\text{ B}$ 时，$\text{Bytes}_{\text{MTP\_clone}} = 3,707,232,256\text{ B} = 3.707\text{ GB}$（$3.453\text{ GiB}$）。

### 合成实验与验证结果（`experiments/torchtitan_kimi_k3_mxfp4_adapter_fsdp`）

运行 `python experiments/torchtitan_kimi_k3_mxfp4_adapter_fsdp/run.py --check` 的确定性验证结果如下：

| 验证维度 | 未修复 / 原始基线行为 | 修复后契约行为 (`#5337` / `#5065` / `#5560`) | 确定性验证结论 |
| :--- | :--- | :--- | :--- |
| Kimi K3 KDA `A_log` / `dt_bias` 往返 | 直接在分片 `DTensor` 上 `pad`/`reshape(-1)` 破坏分片坐标 | 先 `Replicate()` 再 `pad([64]->[128])` / `reshape(8192)`，`from_hf` 后恢复 `Shard(0)` | `a_log_exact_roundtrip = True`, `dtensor_layout_restored = True` |
| DeepSeek-V3 `safetensors` 索引过滤 (`mtp=0`) | 残留 `46,596` 个未写出键，合并产生 `dtype: ""` 损坏文件头 | 精确保留 `45,395` 个键（`mtp=1` 为 `46,183` 个键），空类型头数量为 `0` | `filtered_dcp_consolidation_empty_dtype_entries_mtp0 = 0` |
| DeepSeek-V3 MTP 共享权重导出 (`671B BF16`) | 共享 `tok_embeddings`/`lm_head` Storage 触发 `safetensors` 别名拒绝 | `.clone()` 分配独立存储（`3.707 GB`），`from_hf` 过滤副本 | 往返键值与 Storage 隔离完全一致 |
| `SimpleFSDP` 重复 `data_parallel()` (`meta` 设备) | 第 2 遍扫描触发 `_build_operands()` 报 `weight must be contiguous CUDA storage` | `with disable_active_parametrization():` 下 `_build_operands_calls = 0` | `passes_completed = 2`, `error_raised = None` |

### 架构师的设计练习与可证伪问题

1. **排障练习**：若在训练后导出自定义的 FP8/MXFP4 模型到 HuggingFace Safetensors 时遇到 `safetensors_rust.SafetensorError: Error while deserializing header: unknown variant ''`，应首先检查哪个数据结构？
   - *解答*：检查传给 `HuggingFaceStorageWriter` 的 `fqn_to_index_mapping` 是否包含了由 `model.safetensors.index.json` 继承而来、但 `to_hf()` 实际未写出的量化 Scale 键（如 `.weight_scale_inv` 或 `.weight_scale`）或裁剪掉的层（如 MTP 层、视觉塔）。
2. **可证伪假设**：对于 `SimpleFSDP` 在 `meta` 设备上触发的 `ValueError: weight must be contiguous CUDA storage`，在 `dist_moe` 入口处对权重调用 `.contiguous()` 能否修复该问题？
   - *验证方式*：不能（已在 CI Run `39374680382` 证伪）。因为 `meta` 张量的 `.contiguous()` 依然位于 `meta` 设备（`weight.is_cuda == False`），根因是参数布局检查阶段错误激活了计算参数化钩子，必须通过 `disable_active_parametrization()` 解决。
