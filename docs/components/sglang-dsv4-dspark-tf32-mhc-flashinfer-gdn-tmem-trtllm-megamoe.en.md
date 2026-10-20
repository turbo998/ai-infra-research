# SGLang DeepSeek-V4 DSpark TF32 mHC Row Threshold, FlashInfer Blackwell GDN TMEM Load Fencing, and TensorRT-LLM MegaMoE CuteDSL TEP Global Routed Output Contract

> **中文版**: [sglang-dsv4-dspark-tf32-mhc-flashinfer-gdn-tmem-trtllm-megamoe.md](sglang-dsv4-dspark-tf32-mhc-flashinfer-gdn-tmem-trtllm-megamoe.md)  
> **Evidence Class**: `implementation` (first-hand upstream commits) + `own-offline` (`experiments/sglang_dsv4_mhc_flashinfer_gdn_trtllm_megamoe/run.py --check`)

## 1. First-Hand Source Code & Root-Cause Mechanics

1. **SGLang DeepSeek-V4 DSpark Draft-Stage SM100 TF32 mHC Row Threshold & Post-Load Weight Cache Refresh** (`sgl-project/sglang` commit [`c2e574246485a055c0e51d60d1eae027e5d629c0`](https://github.com/sgl-project/sglang/commit/c2e574246485a055c0e51d60d1eae027e5d629c0), `2026-10-20`):
   - **Code Path**: `python/sglang/srt/models/deepseek_v4_mhc.py` (`_mix_stats_impl`) and `python/sglang/srt/models/deepseek_v4_dspark.py` (`load_weights` / `post_load_weights`).
   - **Mechanism**: Previously, `_mix_stats_impl` only allowed medium row counts (`32 <= rows < 4096`) on Hopper (`is_sm90`), forcing Blackwell (`is_sm100`) to require `x_flat.shape[0] >= 128`. Because DeepSeek-V4 DSpark speculative draft stages typically run at batch row counts of `32..96`, SM100 fell back to 4 separate PyTorch kernels. Commit `c2e57424` lowers the unified SM90/SM100 threshold to `x_flat.shape[0] >= 32` and adds `post_load_weights()` in `deepseek_v4_dspark.py` to call `stage.refresh_mhc_norm_weight_cache()` across all draft stages.

2. **FlashInfer Blackwell Gated Delta Net (`gdn`) TMEM Load Fencing Before `cg1` Handle Release** (`flashinfer-ai/flashinfer` commit [`80a44189a0972922a98317612647df4f3012be74`](https://github.com/flashinfer-ai/flashinfer/commit/80a44189a0972922a98317612647df4f3012be74), `2026-10-20`):
   - **Code Path**: `flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py` and `gated_delta_net_cp_prefill.py` (`compute_group_1_chunk`).
   - **Mechanism**: On Blackwell SM100, `tTR_rKS` and `tTR_rNv` loads from Tensor Memory (`TMEM`) execute asynchronously. Releasing `ks_handle.release()` or `nv_handle.release()` without an explicit TMEM load fence allows the producer UMMA warp to overwrite the shared accumulator stage while the consumer's `tcgen05.ld` is still in flight. Commit `80a44189` inserts `cute.arch.fence_view_async_tmem_load()` immediately before both handle releases.

3. **TensorRT-LLM MegaMoE CuteDSL TEP Global Routed Output Contract** (`NVIDIA/TensorRT-LLM` commit [`46d9966b54d9d11998b712e732fb20f64d4d35c0`](https://github.com/NVIDIA/TensorRT-LLM/commit/46d9966b54d9d11998b712e732fb20f64d4d35c0), `2026-10-20`):
   - **Code Path**: `tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_cute_dsl.py`, `configurable_moe.py`, `interface.py`, and `modeling_kimi_linear.py`.
   - **Mechanism**: `MegaMoECuteDsl` (`MoESchedulerKind.FUSED_COMM`) returns an already globally combined routed output on every rank. Under TEP (`attention TP + MoE EP`), model wrappers that apply an external TP AllReduce would reduce the routed output twice ($TP\times$ inflation). Commit `46d9966b` introduces `accepts_global_routed_output` and `returns_global_routed_output` so opt-in models skip the duplicate AllReduce while generic callers receive a rank partial from `FusedCommMoEScheduler`.

## 2. Deterministic Verification (`experiments/sglang_dsv4_mhc_flashinfer_gdn_trtllm_megamoe`)

| Sub-Track | Metric | Legacy | Patched | Architectural Impact |
|---|---|---:|---:|---|
| SGLang DSpark SM100 TF32 mHC (`c2e57424`) | Fast-path hits across batch rows `16..256` | `2 / 7` (`>=128`) | `6 / 7` (`>=32`) | Latency drops `181.21 us -> 91.18 us` (**`1.9874x` speedup**); stale stage caches `4 -> 0` |
| FlashInfer GDN TMEM Fence (`80a44189`) | Async TMEM overwrite hazards (`8 chunks x 2 handles`) | `16 / 16` | `0 / 16` | `fence_view_async_tmem_load()` eliminates all premature `shared_acc` stage overwrites |
| TRT-LLM MegaMoE TEP (`46d9966b`) | Combined output error under $TP=4, EP=4$ | `+3.75` ($4\times$ routed inflation) | `0.0` | `accepts_global_routed_output` contract prevents duplicate AllReduce across all models |
