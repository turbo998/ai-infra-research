# SGLang MI35x Mooncake `ibv_reg_mr` Disaggregation & Tool Coercion, TensorRT-LLM FP8 Custom-Mask Attention, and vLLM Rust CLI Parity

> **Phase 52 (`priority: 61`) · Track 2 First-Hand Source Analysis**  
> **Upstream Commits**: `sgl-project/sglang@7e0e259435a4` (`#45519`), `a2e4ffbcea18` (`#43941`), `6b6c61314d0c` (`#43698`); `NVIDIA/TensorRT-LLM@5f90e8939d6b` (`#20387`), `1c7fadcfbe50` (`#20253`); `vllm-project/vllm@2916b4b4a6b7` (`#62822`), `bc4794ef9a9f` (`#62825`), `8a21d0a328bb` (`#62416`)  
> **Deterministic Reproduction**: `experiments/sglang-mi35x-ibv-tool-coercion-trtllm-fp8-custom-mask-vllm-rust-cli/run.py --check`

---

## 1. SGLang MI35x (`gfx950`) Mooncake HIP dmabuf Bypass (`#45519`), Whole-Number Float Coercion (`#43941`), and Flux ControlNet SP (`#43698`)

1. **AMD MI350X/MI355X (`gfx950`) Disaggregation OOM/Segfault Fix (`#45519`)**:
   - On ROCm 7.0 with `gfx950`, Mooncake's default `hipMemGetHandleForAddressRange` (HIP dmabuf) registration of large KV cache pools exhausts ROCm driver resources, causing subsequent `hipStreamCreateWithPriority` calls to fail with `HSA_STATUS_ERROR_OUT_OF_RESOURCES` and segfault.
   - `sglang@7e0e259435a4` sets `os.environ.setdefault("MOONCAKE_DISABLE_HIP_DMABUF", "1")` when `_is_hip and get_HIP_arch() == "gfx950"`, routing KV cache registration through standard Verbs `ibv_reg_mr`.
2. **Whole-Number Float Coercion for Integer Tool Parameters (`#43941`)**:
   - Models frequently emit strings like `"3.0"` or `"1e2"` for JSON Schema `"integer"` parameters. When `int(val)` raises `ValueError`, SGLang now parses `f = float(val)` and coerces `int(f)` iff `math.isfinite(f) and f.is_integer()`, while preserving fractional strings (`"3.5"`) and rejecting `"inf"`/`"NaN"`.
3. **Flux DiT ComfyUI ControlNet Residual Deduplication & SP Sharding (`#43698`)**:
   - Deduplicates shared residual tensors by `id(residual)` before dtype/device transfer and shards them via `shard_latents_for_sp` when sequence parallelism is active.

---

## 2. TensorRT-LLM FP8 Custom-Mask Context Attention (`#20387`) & Warmup Cleanup (`#20253`)

- **Tile-Level Dequantization & Clamped FP8 Output (`#20387`)**:
  - Enables `QuantMode.has_fp8_kv_cache()` and `torch.float8_e4m3fn` outputs in `TritonCustomMaskFmha`;
  - Dequantizes paged K/V tiles on the fly in `_fwd_kernel` (`k = (k.to(tl.float32) * kv_scale).to(q.dtype)`) without allocating a BF16 copy of the KV pool, and clamps scaled FP8 outputs to `[-448.0, 448.0]` via `tl.minimum(tl.maximum(result, -448.0), 448.0)`.
- **Obsolete Warmup Switch Removal (`#20253`)**:
  - Removes the test-only `prepare_resource` switch across `KVCacheManager` and drops redundant `max_spec_graph` capture during draft model warmup.

---

## 3. vLLM Rust `return_token_ids` Propagation (`#62822`), Stats Interval Guard (`#62825`), and `ModelArgs` (`#62416`)

1. **`return_token_ids` Propagation (`#62822`)**: Threads `return_token_ids` through Rust `render_chat` and `render_completion` into `GenerationRequest`, populating top-level `prompt_token_ids` and per-choice `token_ids`.
2. **`VLLM_LOG_STATS_INTERVAL` `1ns` Floor & Request ID Determinism (`#62825`)**: Clamps sub-nanosecond positive intervals via `interval.max(Duration::from_nanos(1))` to prevent `tokio::time::interval(0)` panics, and honors `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`.
3. **Rust `ModelArgs::file_sources()` (`#62416`)**: Promotes `--tokenizer`, `--tokenizer-revision`, and `--hf-config-path` from `EngineUnsupportedArgs` into `ModelArgs`.
