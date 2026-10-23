# TensorRT-LLM KVCacheManagerV2 首尾混合精度 NVFP4 KV Cache、MXFP8 E8M0 次正规数 NaN 修复、vLLM V2 投机解码 RoPE 钳位与 SGLang 预填充心跳恢复深度解析

- **研究日期**：2026-10-24（Phase 58 · Priority 67 · Track B）
- **上游证据链**：
  - `NVIDIA/TensorRT-LLM@b4bfec4baf` (`#20452`, *feat(kv_cache): mixed-precision KV cache in KVCacheManagerV2*)
  - `NVIDIA/TensorRT-LLM@348347ce3e` (`#20075`, *[Bugfix] Fix MXFP8 quantization NaN on subnormal E8M0 scale and support Ray named process groups*)
  - `NVIDIA/TensorRT-LLM@96de4749c5` (`#20688`, *perf(speculative): compile _build_replay_work_items_torch*)
  - `NVIDIA/TensorRT-LLM@0f3e1d093d` (`#20276`, *feat(qwen): Qwen3.8-Flash-Next modeling_v2 & NVFP4 KV threshold*)
  - `vllm-project/vllm@37cf712552` (`#63251`, *[Bugfix][V2][SpecDecode] Keep draft model within its RoPE cache on MRV2*)
  - `vllm-project/vllm@44302539f8` (`#55743`, *[Bugfix] Preserve 64-bit integer precision in coerce_to_schema_type*)
  - `vllm-project/vllm@ccd4e4030f` (`#63318`, *feat(frontend): add --return-tokens-as-token-ids to Rust CLI*)
  - `sgl-project/sglang@16aa816060` (`#45922`, *[PD] Reset heartbeat failure count when prefill node fails*)
  - `sgl-project/sglang@30dc2f9430` (`#45982`, *feat(bench): add --input-len-list to bench_one_batch_server*)
- **证据等级**：`implementation` + `own-offline`（`experiments/trtllm-kvcm2-mixed-nvfp4-mxfp8-e8m0-vllm-spec-rope-sglang-hb/run.py` 确定性验证）

---

## 1. 架构背景与核心问题

在长上下文推理与 PD 分离服务中，Phase 58 Track B 解决了四个关键正确性与容量瓶颈：

1. **全量 NVFP4 KV Cache 对 Attention Sink 与近邻窗口的精度退化（`TensorRT-LLM #20452`）**：长序列将全部历史压缩为 NVFP4虽能大幅省显存，但序列开头的 Attention Sink（前 `first_n` 个 token）与最近生成的局部窗口（最后 `last_n` 个 token）对量化噪声高度敏感。如何在不重写所有底层 FlashInfer / TRTLLM FMHA 内核的前提下，在 `KVCacheManagerV2` 中实现页粒度的首尾高精度保留与中间历史 NVFP4 存储？
2. **MXFP8 激活量化在近零非零块上的 `0 * inf = NaN` 静默污染（`TensorRT-LLM #20075`）**：当 MoE 门控激活（如 SwiGLU）产生极小但非零的 32 元素块（`amax in (0, 448 * 2^-127]`）时，其 E8M0 缩放因子被编码为最小指数 `2^-127`。在 IEEE-754 FP32 中，`2^-127 < 2^-126` 属于**次正规数（subnormal）**，而 GPU 硬件指令 `rcp.approx.ftz.f32` 会将次正规输入直接冲刷为 `0.0` 并返回 `+inf`，导致块内 `0.0 * (+inf) = NaN` 扩散至整层。
3. **vLLM Model Runner V2 (MRV2) 投机解码 Draft 模型 RoPE 越界（`vllm #63251`）与 64 位整数工具参数精度丢失（`vllm #55743`）**：当 Draft 模型的 `max_position_embeddings`（如 `8,192`）小于 Target 模型的 `max_model_len`（如 `32,768`）时，MRV2 无法像 V1 那样对超长序列动态跳过投机，导致位置索引越界读取 `cos_sin_cache`；此外工具调用解析器对 `"number"` schema 先调 `float(value)` 会截断 `> 2^53` 的雪花 ID。
4. **SGLang PD 分离架构下 Prefill 节点重连即被误杀（`sglang #45922`）**：`CommonKVManager._handle_node_failure` 在踢出失联 Prefill 节点时未清理 `self.heartbeat_failures[addr]`（残留计数 `3`），导致节点恢复重连后只要遇到哪怕 1 次偶发心跳超时（`3 + 1 >= MAX_HEARTBEAT_FAILURES(3)`）就会立即再次被判定下线。

---

## 2. 核心实现机制与源码走读

### 2.1 TensorRT-LLM `KVCacheManagerV2` 首尾混合精度 NVFP4 KV Cache（`#20452`）

`tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`、`mixed_precision_staging.py` 与 `kernels/blackwell/mixed_precision_kv.py` 协同实现了 `dtype: mixed_nvfp4_bfloat16 | mixed_nvfp4_fp8 | mixed_nvfp4_float16`：

- **`_NVFP4Codec.row_bytes(head_dim)` 精确字节公式**：
  $$\text{row\_bytes}(d) = \left\lfloor \frac{\frac{d}{16} \times (8 + 1) + 7}{8} \right\rfloor \times 8$$
  对于 `head_dim = 128`，每 16 个元素打包为 `8 B` E2M1 + `1 B` FP8 缩放因子，共 `8 * 9 = 72 B`（对比 BF16 的 `256 B`，单行压缩比 **`3.5556x`**；对比 FP8 的 `128 B`，压缩比 **`1.7778x`**）。
- **双池映射与 Staging 页空间**：
  - 每个全注意力层拥有两个逻辑池：主池存放 NVFP4 历史页，追加在模型自身 layer group 之后的 `HIGH_PRECISION` 池存放首部 `mixed_precision_first_n`（如 256）和尾部 `mixed_precision_last_n`（如 256）个 token 所在的页（`pool_ratio: [0.75, 0.25]`）。
  - 在 `with attn_metadata.mixed_precision_kv_staging(layer_idx):` 上下文管理器内，Blackwell CuTe DSL 内核将 NVFP4 历史页反量化并连同高精度页 (`_copy_native` `cute.autovec_copy` 保留位级精确) 一起汇聚到临时 staging 页空间，使 FlashInfer 与 TRTLLM FMHA 内核无需感知混合存储；层执行完毕后再把新生成的 token 写回对应池，并在滑出 `last_n` 窗口时自动降级量化为 NVFP4。
- **容量收益（`head_dim=128, num_kv_heads=8, page_size=32, first_n=256, last_n=256`）**：
  - `seq_len = 4,096`（128 页，保留 16 页 BF16 + 112 页 NVFP4）：单层单序列 KV 显存从 `1,024 KiB` 降至 `380 KiB`（节省 **`62.89%`**）；
  - `seq_len = 32,768`（1,024 页，保留 16 页 BF16 + 1,008 页 NVFP4）：单层单序列 KV 显存从 `8,192 KiB` 降至 `2,396 KiB`（节省 **`70.75%`**）。

### 2.2 TensorRT-LLM `cvt_warp_fp16_to_mxfp8` E8M0 次正规数 `exp2f_rcp` 修复（`#20075`）

在 `cpp/tensorrt_llm/kernels/quantization.cuh` 中，旧代码先将 `__nv_fp8_e8m0` 转回 `float`，再调用 `reciprocal_approximate_ftz(SFValue)`：
```cpp
// 修复前：当 fp8SFVal == 0 (即 2^-127，FP32 次正规数) 时，rcp.approx.ftz.f32 将输入冲刷为 0 -> 返回 +inf
// 块内原本极小或为 0 的元素乘以 +inf 立即产生 0 * inf = NaN！
// 修复后：直接从无偏指数位计算精确的 2^(127 - fp8SFVal)：
float outputScale = vecMax != 0.f ? exp2f_rcp(fp8SFVal) : 0.0f;
```
由于 `2^{+127}` 在 IEEE-754 FP32 中是完全合法的最大正规幂（`~1.70141e38`），任何落入 `(0, 448 * 2^-127]`（即 `(0, 2.6331e-36]`）的近零 SwiGLU 激活块都能无 NaN 地完成量化。

### 2.3 vLLM V2 投机解码 RoPE 缓存二分钳位（`#63251`）、64 位整数保真（`#55743`）与 SGLang 心跳重置（`#45922`）

1. **vLLM `_CACHE_ONLY_ROPE_TYPES` vs `_limit_max_model_len_to_draft()`（`#63251`）**：
   - 若 Draft 模型的 RoPE 类型属于 `_CACHE_ONLY_ROPE_TYPES = {"default", "linear", "llama3", "ntk", "proportional"}`（位置参数仅决定预计算 `cos_sin_cache` 长度而不改变频率公式），`SpeculativeConfig.draft_shares_target_positions()` 将 Draft 的 `max_position_embeddings` 直接抬升至 Target 的 `max_model_len`。
   - 若 Draft 使用依赖上下文长度缩放系数的 RoPE（如 `yarn`, `deepseek`, `dynamic`, `longrope`），则通过 `_limit_max_model_len_to_draft()` 将 V2 引擎的 `max_model_len` 钳位至 `draft_max_model_len`，彻底杜绝 MRV2 CUDA Graph / Eagle 路径上的 OOB rotary 索引读取。
2. **vLLM `coerce_to_schema_type` 整数优先解析（`#55743`）**：
   - 对 JSON Schema `"number"` 类型先尝试 `int(value)` 再回退 `float(value)`，使 `"9007199254740993"`（`2^53 + 1`）保持精确整数而非被 IEEE-754 双精度浮点数截断为 `9007199254740992`。
3. **SGLang `_handle_node_failure` 心跳计数器清理（`#45922`）**：
   - 在 `python/sglang/srt/disaggregation/common/conn.py` 的 `_handle_node_failure(failed_bootstrap_addr)` 中增补 `self.heartbeat_failures.pop(failed_bootstrap_addr, None)`，确保重连后的 Prefill 节点重新获得完整的 `MAX_HEARTBEAT_FAILURES = 3` 容错窗口。

---

## 3. 架构师设计练习与可证伪问题

1. **定量练习**：使用 `_NVFP4Codec.row_bytes(head_dim)` 公式计算 `head_dim = 64, 128, 256` 时的单头行字节数（分别为 `40 B, 72 B, 144 B`），验证为何 `head_dim = 64` 时因 8 字节对齐需要从 `36 B` 补齐至 `40 B`。
2. **可证伪假设**：在 `cvt_warp_fp16_to_mxfp8` 中，若输入块含 1 个等于 `1e-38` 的元素和 31 个 `0.0` 元素，`vecMax != 0.f` 分支被触发且 `fp8SFVal == 0`（`2^-127`）；旧版 `rcp.approx.ftz.f32` 会使那 31 个 `0.0` 元素全部变成 `NaN`，而 `exp2f_rcp(0)` 输出 `0.0`。
