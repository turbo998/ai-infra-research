# SGLang 统一 Radix-Tree 驱逐动作传递与 DeepSeek-V4.1 视觉 MoE 路由仿真、TRT-LLM 基准历史截断及 vLLM 多模态/XPU/Rust 契约深挖

> **一手源码溯源** (`2026-10-21`)：
> - `sgl-project/sglang@5e79333c95c162abf079548fe40a6a2bca6c5409` (`#45439`): *[Radix Tree] Refactor custom tree core (#45439)*
> - `sgl-project/sglang@ed268dd32097ed44a83b260250defb82256602e2` (`#45628`): *Fix routing simulation for DeepSeek-V4.1 vision-biased MoE gates (#45628)*
> - `NVIDIA/TensorRT-LLM@7f0caf443ec440ffc7ca455c91ff6f7c8b0b8d67` (`#20297`): *[https://nvbugs/6925906][infra] Floor gen_only device step time perf history at metric cutover (#20297)*
> - `vllm-project/vllm@65a886c69f44cf446d6afa03315f39590ccd843e` (`#62859`): *[Bugfix] Reject multimodal inputs when chunked processing is enabled (#62859)*
> - `vllm-project/vllm@35d01660ff3644de128253950b8fcb4b7705668f` (`#62244`): *[XPU] Enable speculative decoding adaptive verification without confidence head on XPU (#62244)*
> - `vllm-project/vllm@18e8791134e8e46134f982c8263d5fa0a50548c0` (`#62823`): *[Rust Frontend] Apply response_format over structured_outputs like Python (#62823)*

---

## 1. SGLang 统一 Radix-Tree 驱逐动作传递 (`#45439`) 与 DeepSeek-V4.1 视觉 MoE 路由仿真 (`#45628`)

### 1.1 跨 Python 与 Rust 的组件化 Radix-Tree 驱逐契约 (`#45439`)

在混合注意力/状态模型（包含 Full KV、Mamba 状态、SWA 滑动窗口及 `C128SidecarComponent` 旁路块）的树缓存中，旧实现把节点驱逐硬编码为 `device_frees` 与 `host_frees` 两组槽位列表，导致自定义组件在设备驱逐、主机晋升、级联驱逐或窗口锁释放（`release_window_lock`）时产生的 `ComponentAction` 旁路动作无法穿透统一核心层。`#45439`（横跨 31 个 Python/Rust 文件，`+3,379 / -2,027` 行）完成了三项架构重构：
1. **统一 `evict_component` 传递链**：在 Python `CustomRadixTreeCore` 与 Rust `sglang-radix-tree` (`src/core/evict.rs`, `src/core/insert.rs`) 中，将 `cache_actions: list[CacheAction | ComponentAction]` 贯穿 `evict_device_next_node`、`drive_host_eviction`、`_cascade_evict` 与 `release_window_lock`。
2. **`BaseEvictionResult.__del__` 防泄漏 Tripwire**：在 `BaseEvictionResult` 析构函数中强制断言：
   ```python
   assert (
       not self.device_frees
       and not self.host_frees
       and not self.cache_actions
   ), "EvictionResult destroyed with undrained resources"
   ```
   任何调用方若未通过 `drain_device_frees()` / `drain_host_frees()` / `drain_cache_actions()` 消费全部释放槽位与组件动作，将在对象销毁时立即触发确定性异常。

### 1.2 DeepSeek-V4.1 视觉偏置 MoE 门控的均衡路由仿真修复 (`#45628`)

DeepSeek-V4.1 的视觉偏置 MoE 门控走独立的 `vl_routing.py` (`_finish_vision_topk`) 路径，此前绕过了 `CustomTopK` 的 `SGLANG_SIMULATE_UNIFORM_EXPERTS` 与 `SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS` 环境变量，导致多模态负载在做满载通信压测时专家负载依然发生偏斜。`#45628` 将仿真逻辑提取为共享模块 `sglang/srt/layers/moe/topk_simulation.py` 并接入 `_finish_vision_topk`，解决了三个关键边界问题：
1. **保护融合共享专家槽位**：仅对路由专家槽位 `num_routed_topk = top_k - num_shared_topk_slots` 重写专家 ID，保留末尾 `num_shared_topk_slots` 个融合共享专家槽位不被覆盖。
2. **序列切分 Rank 偏移**：当 `moe_input_scattered=True`（开启 `deepep` 或 `flashinfer` Token 切分分发）且 `attn_tp_sequence_sharded=True` 时，精确计算全局 Token 分片序号：
   $$\text{token\_shard\_rank} = \text{attn\_dp\_rank} \times \text{attn\_tp\_size} + \text{attn\_tp\_rank}, \quad \text{num\_token\_shards} = \text{attn\_dp\_size} \times \text{attn\_tp\_size}$$
   避免不同 TP Rank 在 Round-Robin 仿真中生成完全相同的专家序列。
3. **重置 CUDA Graph Padded 行掩码标志**：在 `apply_routing_simulation` 重写 `topk_ids` 后，立即将 `padded_rows_masked = False` 重置，强制后续 `_post_process_topk_ids` 重新将 CUDA Graph 填充位置掩码为 `-1`。

---

## 2. TensorRT-LLM `gen_only` 单步设备耗时历史截断防误报机制 (`#20297`)

在 `ceil(concurrency / dp) > max_batch_size` 的生成压测配置下（例如 `concurrency=64, max_batch_size=32, osl=400`），基准测试被拆分为第一个满 Batch 波次（32 请求，约 `420` 次迭代，单步耗时约 `7.5 ms`）与第二个短尾波次（剩余请求，同样约 `420` 次迭代，因有效 Batch 极小单步耗时仅约 `2.5 ms`，相差 **`~3.0x`**）。
- 在 `2026-10-01` 切分点之前，`gen_only` 采用众数（Modal）`num_generation_tokens` 桶选取稳态波次；当两波次迭代次数接近（如 `420 vs 421`）时，众数桶在满波次与尾波次之间随机跳变，把 `~2.5 ms` 的尾波次混入 90 天历史库。
- 切分点之后，`gen_only` 改为按 `expected_gen_only_ngen` 锁定第一个满 Batch 稳态波次（恒定 `~7.5 ms`）。然而，90 天三日滑动平均后的 **P5 百分位基线**仍被旧的 `~2.5 ms` 记录拉低至 **`3.3117 ms`**，导致正常的 `7.51 ms` 持续触发假阳性回归告警。
- `#20297` 引入 `GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS = 1790856543000` (`2026-10-01T12:09:03Z`)：对 `gen_only` 的 5 个单步耗时指标（`d_{mean,median,std,p75,p99}_gen_worker_per_iter_device_step_time`）过滤掉 `ts_created < 1790856543000` 的历史点并失效化切分前的存储 `d_baseline_*` 锚点（保留 `d_threshold_*` 阈值覆盖及 `e2e` 全历史），使 P5 基线恢复至 **`7.4960 ms`**，彻底消除假阳性报警。

---

## 3. vLLM 多模态分块 Embedding 拦截 (`#62859`)、XPU 投机解码在线估计器 (`#62244`) 与 Rust 结构化输出优先级 (`#62823`)

1. **多模态分块 Embedding 预校验 (`#62859`)**：此前 `PoolingServeContext._maybe_pre_process_chunked` 仅检查 `len(self.prompt_token_ids) == len(self.engine_prompts)`；当开启 `--enable-chunked-processing` 且单个 Embedding Batch 混合了纯文本与多模态输入时，文本项被原地截断为首块，而多模态项保持未分块并流入 `_merge_Install`，导致静默算出错误 Embedding。`#62859` 在修改任何状态前扫描全 Batch，一旦发现 `is_multimodal_raw_input(prompt)` 立即抛出 `VLLMValidationError` (`400 Bad Request`)。
2. **XPU 无置信度头的投机解码自适应验证 (`#62244`)**：在 XPU 平台上启用 `OnlineAcceptanceEstimator`（带阻尼因子 $\frac{n}{n + N_{\text{damp}}}$、$N_{\text{damp}}=20$ 的 $O(K)$ 箭头矩阵求解器）并在 `XPUPlatform.get_attn_backend_cls` 中对 `use_spec_decode` 保持 `AttentionCGSupport.ALWAYS`。
3. **Rust OpenAI Server `response_format` 覆盖与冲突校验 (`#62823`)**：对齐 Python `structured_outputs_from_response_format()` 语义：当 `response_format` 为 `text` 或未指定时保留 `structured_outputs`；当两者指定**同类型 (`discriminant`)** 约束（如均为 `json_schema`）时，以 `response_format` 的约束覆盖并保留 `structured_outputs` 的辅助参数（如 `whitespace_pattern`）；当两者指定**不同类型**约束时立即返回 `400 Bad Request`。
