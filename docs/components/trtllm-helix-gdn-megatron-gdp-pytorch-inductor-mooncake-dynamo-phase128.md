# Phase 128 深度研究（Track C）：TensorRT-LLM Helix CP 预上投影与 GDN 零拼接、Megatron-LM GDP FP32 梯度保留与全局批次切换、PyTorch Inductor 拼接融合与 64 位池化及 Mooncake/Dynamo 契约

## 1. 背景与核心问题

在大规模上下文并行（Helix CP）推理、Gated Delta Product（GDP）与混合 Mamba 训练/推理、PyTorch Inductor 算子融合以及分布式 KV 传输运行时（Mooncake TENT / NVIDIA Dynamo）中，微小的代理排序遗漏、低精度激活饱和、整数索引溢出或默认控制面连接都会引发数据竞争、梯度消失、非法显存访问或连接风暴。本报告基于 2026-10-28 合入 `NVIDIA/TensorRT-LLM`（3 个）、`NVIDIA/Megatron-LM`（2 个）、`pytorch/pytorch`（2 个）、`kvcache-ai/Mooncake`（1 个）与 `ai-dynamo/dynamo`（2 个）的 10 个上游提交展开深度剖析。

## 2. 关键实现路径与机制剖析

### 2.1 TensorRT-LLM Helix CP 全交换前 MLA V 上投影、FIFO 异步围栏、Eager GDN 零拼接与 Beam Search 跳过单位温度缩放（`b613fbe387ed`, `99466ea9ad97`, `3b4eebcf020e`）
1. **Helix CP MLA V 预上投影与 FIFO 异步共享内存围栏（`b613fbe387ed`）**：
   - 在 Helix CP 中，MLA 部分输出原本以 `kv_lora_rank = 512` 维进行跨 CP Rank All-to-All 交换，随后再做 V 上投影。新路径通过 `helix_v_b_proj_full`（`[num_heads_tp, v_head_dim, kv_lora_rank]`）在 CP All-to-All 之前对本 Rank 部分输出执行 BF16 V 上投影。当高 TP 下每 Rank 头数较少（例如 128 头在 `TP=64` 下 `num_heads_tp = 2`，`2 * 128 = 256 < 512`）时，All-to-All 传输载荷直接减半（**`2.0x` 缩减**）；同时 `Request.get_input_token_ids_array()` 以单次 `memcpy` 暴露 NumPy `int32` 数组，使 `partition_context_for_helix` 达到 `O(prompt/cp)` 复杂度。
   - 在 `helixAllToAllKernel` 中，下一轮循环通过异步代理（Async Proxy）执行 `cp.async.bulk` 重填共享内存，而当前轮的 `sanitize/protoPack` 存储与 S2G 读取走通用代理（Generic Proxy）；新增 `fence_view_async_shared()` 强制跨代理视图有序，消除共享内存覆盖竞争。
2. **Eager GDN 共享输出缓冲区零拼接（`99466ea9ad97`）**：在 `Qwen3NextGatedDeltaNet.forward` 的 Eager 混合批次执行中预分配单块共享 `attn_out`，使 Prefill 与 Decode 直接写入各自的不相交切片，将额外输出 `torch.cat` 拷贝次数降为 **`0`**。
3. **Beam Search 单位温度跳过全词表缩放（`3b4eebcf020e`）**：当所有请求 `temperature == 1.0` 时保持 `temperature = None`，跳过对全词表 Logits 的无意义除法遍历。

### 2.2 Megatron-LM GDP `beta` FP32 Sigmoid 梯度保留与全局批次计划切换检测（`a296523510e1`, `d1eec51b8563`）
1. **GDP `beta` FP32 Sigmoid 保留饱和区梯度（`a296523510e1`）**：在 `GatedDeltaProductMixer._compute_gating` 中将 `b.sigmoid()` 改为 `b.float().sigmoid().to(b.dtype)`。在 BF16（7 位尾数，1.0 附近步长 `2^-7 = 0.0078125`）下，当门控输入 `b = 8.5` 时，`1 - sigmoid(8.5) ≈ 0.0002034 < 0.00390625`，BF16 前向输出直接舍入为 `1.0`，导致反向导数 `sigmoid * (1 - sigmoid)` 变为 **`0.0`（梯度完全消失）**；而在 FP32 中计算并保存 Sigmoid 激活可精确保留 **`0.00020339`** 的非零梯度。
2. **按全局批次大小检测分步 Batch-Size 计划切换（`d1eec51b8563`）**：当 `step_batch_size_schedule` 将全局批次从 `8,192` 调至 `12,288` 且 `DP = 8,192`、`micro_batch_size = 1` 时，未校验的 `num_microbatches = floor(12288 / 8192) = 1` 与旧值 `1` 相等，导致仅比较 `get_num_microbatches()` 漏检批次切换。新增 `is_batch_size_transition` 同时比对 `get_current_global_batch_size() != global_batch_size`，确保 100% 捕获非整除跳变。

### 2.3 PyTorch Inductor `pointwise_cat` 内部消费者排除与 CUDA `max_pool3d` 64 位切片索引（`dd50eb629ec0`, `53765f04b1b4`）
1. **Inductor `pointwise_cat` 内部消费者排除（`dd50eb629ec0`）**：对于 `rotated = sin(x); negated = -rotated; cat((rotated, negated))`（RoPE 典型模式），`rotated.users = {negated, cat}`。旧启发式仅排除当前 `cat` 节点，将同一 `cat` 的另一直接输入 `negated` 误判为外部消费者，从而禁用 `pointwise_cat` 并回退至 `ConcatKernel`（产生 `reinterpret_tensor` 与额外显存落地）。修复后构建 `cat_input_nodes = OrderedSet(input_nodes)`，将属于同一 `cat` 输入集合的消费者视为内部依赖，恢复单核融合。
2. **CUDA `max_pool3d` 64 位单切片索引（`53765f04b1b4`）**：`max_pool3d_with_indices_single_out_frame` 此前以 32 位 `int` 计算单空间切片 `T * H * W` 内的元素偏移 `t * iheight * iwidth + h * iwidth + w`。当单切片超过 `2^31 - 1` 元素（如 `2048 × 1024 × 1024 = 2,147,483,648`）时，32 位有符号整数回绕为负数 `-2,147,483,648`，引发越界前读与 CUDA 非法显存访问（ROCm `HSA_STATUS_ERROR_MEMORY_FAULT`）。提升为 `(int64_t(t) * iheight + h) * iwidth + w` 后彻底消除溢出。

### 2.4 Mooncake TENT 指标自观测契约与 Dynamo 按需 NATS 连接及停止流释放（`d3ef7b1c2821`, `1b6416a308ee`, `78ac1bc99e0a`）
- **Mooncake TENT 共享 DSO 单例与自观测契约（`d3ef7b1c2821`）**：将 `libtent_metrics` 编译为共享库，消除同一进程同时加载 `engine.so` 与 `store.so` 时产生 **2 个独立 `TentMetrics` 单例**与 `9100` 端口冲突的问题（收敛为 **1 个进程级单例**）；当 Prometheus/JSON 序列化失败时返回 HTTP `500` 并在下次成功抓取时暴露失败计数。
- **Dynamo 按需连接 NATS（`1b6416a308ee`）与本地停止后立即释放 vLLM 流（`78ac1bc99e0a`）**：移除仅因环境变量存在 `NATS_SERVER` 就强制初始化 NATS 客户端的逻辑，仅当 `request_plane` 或 `event_plane` 实际使用 NATS 时才连接（TCP+ZMQ 模式下空闲 NATS 连接从 `1 -> 0`）；同时在前端 `vllm_processor` 触发本地停止（Local Stop）后立即释放底层 vLLM Stream，防止长尾生成流泄漏。

## 3. 确定性实验结果汇总

运行 `python experiments/trtllm_helix_gdn_megatron_gdp_pytorch_inductor_mooncake_dynamo_phase128/run.py`：

| 组件 / 提交 | 验证属性 | 修复前 / 基线 | 修复后 / 新路径 | 核心收益 |
|---|---|---:|---:|---|
| TRT-LLM Helix CP (`b613fbe387ed`) | `TP=64` MLA All-to-All 载荷 / FIFO 围栏 | `512` floats/token / 缺异步围栏 | `256` floats/token + `fence_view_async_shared` | 载荷减半（**`2.0x`**）+ 消除 `cp.async.bulk` 竞争 |
| Megatron GDP (`a296523510e1`) | `b=8.5` 处 BF16 Sigmoid 反向梯度 | `0.0`（舍入至 1.0 梯度消失） | `0.00020339`（FP32 Sigmoid） | 保留饱和区有效门控梯度 |
| Megatron Batch (`d1eec51b8563`) | `DP=8192` 下 `8192->12288` 切换检测 | `False`（`microbatches` 均为 1） | `True`（校验 `global_batch_size`） | 消除大规模 DP 下的批次跳变漏检 |
| PyTorch Inductor (`dd50eb629ec0`) | `cat((rot, -rot))` 外部消费者计数 | `1`（回退 `ConcatKernel`） | `0`（启用 `pointwise_cat`） | RoPE 风格拼接恢复单核融合 |
| PyTorch Pool3D (`53765f04b1b4`) | `2048×1024×1024` 切片索引 | `-2,147,483,648`（int32 溢出崩溃） | `+2,147,483,648`（int64 安全寻址） | 消除 >2G 元素切片非法访存 |
| Mooncake & Dynamo (`d3ef7b1c2821` 等) | TENT 单例数 / TCP+ZMQ NATS 连接 | `2` 个冲突单例 / 误连 `NATS_SERVER` | `1` 个共享 DSO 单例 / `0` 冗余连接 | 消除 9100 端口冲突、NATS 误连与 vLLM 流泄漏 |
