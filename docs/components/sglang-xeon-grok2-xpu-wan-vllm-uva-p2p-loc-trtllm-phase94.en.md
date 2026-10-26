# SGLang Xeon Grok-2 & XPU WanVideo Guard, vLLM By-Reference UVA Staging, Mode-Aware P2P KV Offload Timeouts, CUDA 13.4 Locality-Domain NVFP4 MoE & TensorRT-LLM Disaggregated Responses API (Phase 94)

> **中文版**: [sglang-xeon-grok2-xpu-wan-vllm-uva-p2p-loc-trtllm-phase94.md](sglang-xeon-grok2-xpu-wan-vllm-uva-p2p-loc-trtllm-phase94.md)  
> **Evidence Class**: `implementation` + `own-offline`  
> **Experiment Path**: `experiments/sglang_xeon_grok2_xpu_wan_vllm_uva_p2p_loc_trtllm_phase94/`

---

## 1. Architectural Overview

Phase 94 Track 2 examines ten upstream commits across SGLang (`#46393`, `#46049`, `#46501`, `#43341`), vLLM (`#63633`, `#63781`, `#61685`, `#60968`), and TensorRT-LLM (`#19883`, `#20643`):

1. **SGLang Xeon CPU Grok-2 Enablement (`#43341`)**: Implements AVX-512 FP32-accumulated router dot products (`router_dot_fp32_accum`), `fused_moe_router_cpu` with `tanh` softcapping, `fused_dual_residual_rmsnorm_cpu` with BF16 round-trip before post-residual addition to match CUDA numerics, and sets `weight_start = 0` when `_is_cpu and self.use_presharded_weights` (eliminating an out-of-bounds offset of `6144` on rank 3 with `shard_size=2048`).
2. **SGLang Intel XPU WanVideo Guard, Grammar HTTP 500 Status & HiCache Delegation (`#46393`, `#46049`, `#46501`)**: Skips CuTe DSL `fused_Add_norm_interleave_rope` imports on Intel XPU, attaches `HTTPStatus.INTERNAL_SERVER_ERROR` (`500`) to `FINISH_ABORT` on grammar `accept_token` exceptions, and forwards `on_host_pools_registered()` through the HiCache failure-injector wrapper.
3. **vLLM MRV2 By-Reference UVA Staging (`#63633`)**: Stages `np.ndarray` and `array.array` chunks (`int32: 'i'`, `int64: 'q'`, `float32: 'f'`) by reference in `StagedWriteTensor` and copies directly into the UVA pool without intermediate Python list conversion or `async_tensor_h2d` staging.
4. **vLLM Mode-Aware P2P KV Offload Fail-Fast Timeouts (`#63781`)**: Splits `_LOAD_TIMEOUT_S` into `_P2P_LOAD_TIMEOUT_S = 10.0s` (symmetric probed P2P, `3.0x` faster recovery) and `_PD_LOAD_TIMEOUT_S = 30.0s` (disaggregated PD), tightens `_LOOKUP_TIMEOUT_S` (`30.0 -> 8.0s`, `3.75x`) and `_ABORT_ACK_TIMEOUT_S` (`10.0 -> 5.0s`, `2.0x`), and preserves `_REAPED_ID_RETENTION_S = 60.0s > _PD_LOAD_TIMEOUT_S`.
5. **vLLM CUDA 13.4 Locality-Domain NVFP4 MoE Allocation & Step3 Tool Parser Fix (`#61685`, `#60968`)**: Introduces `LocalizedResources` (`SMPartition`, `GreenContext`, `LocalizedAllocator`) splitting 140 SMs into `[60, 68]` locality domains plus `12` remainder backfill SMs; fixes `<steptml:parameter>` extraction with non-greedy `(.*?)` + `re.DOTALL` so values containing `<` are preserved.
6. **TensorRT-LLM Disaggregated `/v1/responses` & `MuseSparkOpenConfig` (`#19883`, `#20643`)**: Adds stateless `/v1/responses` to `OpenAIDisaggregatedServer` (rejecting `previous_response_id` with HTTP `400` and filtering opening SSE events via `_has_responses_output_event`) and translates composite Muse Spark per-layer RoPE/NoPE lists into `every_n_layers_nope = 4` (`8/32` global NoPE layers).

---

## 2. Deterministic Verification Summary (`own-offline`)

| Framework & PR | Invariant / Metric | Before | After | Impact |
|---|---|---:|---:|---|
| SGLang `#43341` / `#46049` | CPU presharded rank-3 offset / Grammar abort HTTP status | `6144` / `200` | `0` / `500` | Fixes OOB slice & silent abort |
| vLLM `#63633` MRV2 UVA | `int32`/`int64`/`float32` `array` itemsize | Double H2D copy | `4 / 8 / 4 B` direct UVA | Zero intermediate staging tensor |
| vLLM `#63781` P2P KV Offload | Symmetric P2P Load / Lookup / Abort timeouts | `30 / 30 / 10 s` | `10 / 8 / 5 s` | `3.0x` / `3.75x` / `2.0x` faster fail-over |
| vLLM `#61685` / `#60968` | 140-SM partition (`[60, 68]`) / Step3 `<` matches | Unpartitioned / `0` | `12` rem SMs / `2` | L2 locality binding + SQL `<` retention |
| TensorRT-LLM `#19883` / `#20643` | Disagg `previous_response_id` / Muse Spark NoPE layers | HTTP `500` / N/A | HTTP `400` / `8/32` | Explicit stateless API + periodic NoPE |
