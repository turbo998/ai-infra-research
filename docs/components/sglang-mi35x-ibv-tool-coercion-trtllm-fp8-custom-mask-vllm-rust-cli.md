# SGLang MI35x Mooncake `ibv_reg_mr` 分离式推理与工具整数参数转型、TensorRT-LLM FP8 Custom-Mask 注意力及 vLLM Rust CLI 对齐深度解析

> **Phase 52 (`priority: 61`) · Track 2 一手源码深挖**  
> **上游提交**: `sgl-project/sglang@7e0e259435a4` (`#45519`), `a2e4ffbcea18` (`#43941`), `6b6c61314d0c` (`#43698`); `NVIDIA/TensorRT-LLM@5f90e8939d6b` (`#20387`), `1c7fadcfbe50` (`#20253`); `vllm-project/vllm@2916b4b4a6b7` (`#62822`), `bc4794ef9a9f` (`#62825`), `8a21d0a328bb` (`#62416`)  
> **确定性复现脚本**: `experiments/sglang-mi35x-ibv-tool-coercion-trtllm-fp8-custom-mask-vllm-rust-cli/run.py --check`

---

## 1. SGLang MI35x (`gfx950`) Mooncake HIP dmabuf 规避 (`#45519`)、工具整数参数转型 (`#43941`) 与 Flux ControlNet SP 切分 (`#43698`)

1. **AMD MI350X/MI355X (`gfx950`) 分离式推理 OOM 崩溃修复 (`#45519`)**：
   - 在 ROCm 7.0 + `gfx950` 上，Mooncake 默认调用 `hipMemGetHandleForAddressRange` (HIP dmabuf) 注册大块 KV Cache 缓冲区，会耗尽 ROCm 驱动内部的内存句柄资源，导致后续 `hipStreamCreateWithPriority` 报 `HSA_STATUS_ERROR_OUT_OF_RESOURCES` 并触发段错误崩溃。
   - `sglang@7e0e259435a4` 在 `MooncakeTransferEngine.__init__` 检测到 `_is_hip and get_HIP_arch() == "gfx950"` 时，自动设置 `os.environ.setdefault("MOONCAKE_DISABLE_HIP_DMABUF", "1")`，强制回退至标准 Verbs `ibv_reg_mr` 路径注册 KV Cache。
2. **工具调用解析器整数参数整值浮点字符串转型 (`#43941`)**：
   - LLM 常为 JSON Schema 中声明为 `"integer"` 的工具参数生成形如 `"3.0"` 或 `"1e2"` 的字符串。此前 `int(val)` 直接抛出 `ValueError` 并保留原字符串，导致严格类型校验失败。
   - 修复后：在 `int(val)` 失败时尝试 `f = float(val)`，当且仅当 `math.isfinite(f) and f.is_integer()` 时转为 `int(f)`（如 `"3.0" -> 3`、`"-7.00" -> -7`、`"1e2" -> 100`），而小数 `"3.5"` 与非有限值 `"inf"`/`"NaN"` 原样保留以交由下游报错。
3. **Flux DiT ComfyUI ControlNet 残差去重与序列并行切分 (`#43698`)**：
   - `prepare_control_residual_kwargs` 按 `id(residual)` 对跨层共享的 ControlNet 残差张量做转换缓存去重，并在 `batch.did_sp_shard_latents` 开启时调用 `shard_latents_for_sp` 将残差切分为当前 SP Rank 对应的图像 Token 分片；同时拒绝被 `cache-dit` 等包装器篡改长度的 Block 列表。

---

## 2. TensorRT-LLM FP8 Custom-Mask Context Attention (`#20387`) 与 Warmup 路径清理 (`#20253`)

- **Tile 级在线反量化与 FP8 E4M3 饱和截断 (`#20387`)**：
  - `TritonCustomMaskFmha` 放开 `QuantMode.has_fp8_kv_cache()` 与 `torch.float8_e4m3fn` 输出支持；
  - 在 `triton_prefill.py` 的 `_fwd_kernel` 中，直接按 Tile 读取分页 FP8 KV Cache 并执行 `k = (k.to(tl.float32) * kv_scale).to(q.dtype)` 与 `v = (v.to(tl.float32) * kv_scale).to(q.dtype)`，**无需在显存中额外分配整池 BF16 副本**（例如 `128 pages × 16 tokens × 2 heads × 512 dim` 下避免 `8 MiB` 额外拷贝）；
  - 当输出为 `tl.float8e4nv` (`float8_e4m3fn`) 时，乘上 `out_scale` 后执行 `tl.minimum(tl.maximum(result, -448.0), 448.0)` 饱和截断再落盘。
- **移除冗余 Warmup 开关与重复 CUDA Graph 捕获 (`#20253`)**：
  - 清理 `KVCacheManager.add_dummy_requests` 中仅用于测试的 `prepare_resource` 分支，并删除 draft model warmup 阶段重复的 `max_spec_graph` 捕获。

---

## 3. vLLM Rust 前端 Token ID 透传 (`#62822`)、零间隔防 Panic (`#62825`) 与 `ModelArgs` 拆分 (`#62416`)

1. **`return_token_ids` 全链路透传 (`#62822`)**：Rust `render_chat` 与 `render_completion` 将请求中的 `return_token_ids` 写入 `GenerationRequest`；当启用时，流式首包与非流式响应同时返回顶层 `prompt_token_ids` 及每个 Choice 的 `token_ids`。
2. **`VLLM_LOG_STATS_INTERVAL` 零值保护与确定性 Request ID (`#62825`)**：
   - 将正浮点数秒转换为 `Duration` 后执行 `interval.max(Duration::from_nanos(1))`，防止极小正数下取整为 `0ns` 导致 `tokio::time::interval` 触发 Panic；非有限值或负数禁用周期日志。
   - 支持 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`，启用后生成确定性请求 ID（如 `cmpl-req-42-0`）。
3. **Rust CLI 独立 Tokenizer 与 HF Config 源解析 (`#62416`)**：將 `--tokenizer`、`--tokenizer-revision` 与 `--hf-config-path` 从 `EngineUnsupportedArgs` 迁移至 `ModelArgs::file_sources()`，与 Python 端语义完全对齐。
