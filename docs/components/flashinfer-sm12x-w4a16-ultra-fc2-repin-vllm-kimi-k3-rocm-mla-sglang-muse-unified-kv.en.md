# FlashInfer SM12x W4A16 TC-Decode Ultra FC2 Forced-Tile Re-Pin, vLLM ROCm Kimi-K3 GLA Gate + MLA FP8 Prefill Fusion & SGLang Muse Spark Unified KV Memory Guard

**中文版**: [flashinfer-sm12x-w4a16-ultra-fc2-repin-vllm-kimi-k3-rocm-mla-sglang-muse-unified-kv.md](flashinfer-sm12x-w4a16-ultra-fc2-repin-vllm-kimi-k3-rocm-mla-sglang-muse-unified-kv.md)

## 1. Architectural Context & Problem Statement

Across Blackwell SM12x (`GB10` SM121), AMD CDNA4 (`gfx950`), and B200 speculative/unified-memory serving stacks, invariant mismatches across custom-op boundaries and memory pools trigger launch crashes, redundant HBM traffic, or silent KV corruption:
1. **FlashInfer SM12x W4A16 TC-Decode Ultra FC2 Re-Pin Crash**: On `nvidia/Qwen3.6-35B-A3B-NVFP4` (`hidden_size=2048, intermediate_size=512, top_k=8`), small decode batches (`m <= 4`) enable `tc_decode_fused_sum=True` and widen the FC2 tile `128 -> 256 -> 512` to fit in a single persistent wave, which forces `tile_k = 32` at 256 threads. Crossing `torch.ops.flashinfer.w4a16_fused_moe_launch` re-invokes `compile_w4a16_fused_moe` with `force_tile_config`, where `_candidate_tile_fits` enforced the search-space floor `tile_k >= 64` and raised `ValueError: force_tile_config fc2 tile (tile_k=32, tile_n=512) does not fit problem N/K=2048/512`.
2. **vLLM ROCm Kimi-K3 Unfused GLA Gate & MLA FP8 Prefill Prep**: When `mla_use_output_gate=True`, `g_proj` ran as a separate `ColumnParallelLinear` reading `hidden_states` twice; decode padded `24` query heads to `32` via an extra copy; and CDNA4 FP8 prefill executed separate `k_pe` concat, head padding, FP8 cast, and per-step scalar `torch.ones(())` allocations.
3. **SGLang Muse Spark Unified KV Memory & DFLASH FA4 Draft Fusion**: Unified KV pools interleave layers within each token row, breaking `MuseSparkAttention._fused_kv_store` (which assumes contiguous per-layer planes), while `_fused_draft_decision` skipped compile-time dtype validation when `--speculative-draft-kv-cache-dtype` was `auto`/`None`, failing to detect that DFLASH + FA4 always uses the draft compute dtype (`bf16`) against a quantized host KV cache.

## 2. First-Hand Implementation Audit

### 2.1 `flashinfer-ai/flashinfer` (`fc8cfd004324f5f3dbd54ed4aa50192aa7592888`, `#6292`, `2026-10-20`)
- **`flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py`**:
  - Adds `enforce_search_floor: bool = True` to `_candidate_tile_fits` and passes `enforce_search_floor=False` during `force_tile_config` re-validation, checking only divisibility (`problem_n % tile_n == 0`, `problem_k % tile_k == 0`), NVFP4 16-element scale alignment (`tile_k % 16 == 0`), and shared-memory footprint (`smem_bytes <= max_shared_mem`).
  - **Author Benchmark (`author-benchmark`)**: On DGX Spark GB10 (SM121), 5-shot GSM8K (1319 questions) on `nvidia/Qwen3.6-35B-A3B-NVFP4` achieves **`0.9158`** with fixed W4A16 (`0.7.0.post1 + fix`), matching `0.6.16.post3` (`0.9136`) and recovering **`+2.58` points** over `b12x` W4A4 (`0.8900`).

### 2.2 `vllm-project/vllm` (`09fd027217648768c417261ab9a90a3149649ad2`, `#59634`, `2026-10-20`)
- **`vllm/models/kimi_k3/amd/{linear.py,mla.py,mtp.py}` & `rocm_aiter_mla.py`**:
  - Fuses `.q_a_proj` (`1536`), `.kv_a_proj_with_mqa` (`576`), and `.g_proj` (`3072`) into `KimiK3MergedQKVGateLinear` (`5184` output columns), saving **`58,720,256 B` (`56.0 MiB`)** of BF16 activation reads per layer at `4096` tokens;
  - Allocates `q_out_padded` of shape `(num_tokens, 32, 576)`, passes `q_out = q_out_padded[:, :24]`, and recovers the 32-head view in `get_mla_padded_q` via zero-copy `q.as_strided` when `q.stride(0) == m * q.stride(1)`;
  - Enables `fused_mla_prefill_qkv_fp8` on CDNA4 (`gfx950`) when `not any(d & (d - 1) for d in head_dims)` and caches `self._fp8_prefill_one_scale`.

### 2.3 `sgl-project/sglang` (`cfec1b91bb33f3267d1811680b51fd581e0df458`, `#45356`, `2026-10-20`)
- **`python/sglang/srt/models/muse_spark.py` & `mem_cache/kv_cache_configurator.py`**:
  - Returns `None` from `_fused_kv_store` when `get_memory().enable_unified_memory` is True;
  - Evaluates `configure_kv_cache_dtype` in `_fused_draft_decision` whenever `self.spec_algorithm.is_dflash_family() and draft_backend == "fa4"`, declining fusion when draft `bf16` mismatches host `fp8`/`fp4`.

## 3. Deterministic Verification (`own-offline`)

Run `python experiments\flashinfer_sm12x_w4a16_vllm_kimi_rocm_sglang_unified_kv\run.py --check`:

| Subsystem | Metric | Before / Baseline | After (`Phase 37`) | Evidence Tier |
|---|---|---|---|---|
| FlashInfer SM12x W4A16 | Ultra FC2 (`32, 512, 256`) Re-Pin | `False` (`ValueError` crash) | **`True` (rejects `k=8, n=768, smem=1024`)** | `implementation` + `own-offline` |
| Qwen3.6-35B-A3B GSM8K | GB10 (SM121) W4A16 vs W4A4 | `0.8900` (W4A4) | **`0.9158` (`+2.58 pts`)** | `author-benchmark` |
| vLLM ROCm Kimi-K3 MLA | Merged QKV+Gate Cols / Saved Read | 2 GEMMs | **1 GEMM (`5,184` cols, `-58.72 MB`/layer)** | `implementation` + `own-offline` |
| vLLM CDNA4 MLA Prefill | `as_strided` Pad + Fused QKV FP8 | 4 launches + copy | **1 launch + `0 B` copy** | `implementation` + `own-offline` |
| SGLang Muse Spark | Unified KV Store / DFLASH FA4 Fusion | Enabled (corrupts stride/dtype) | **`None` (safe fallback) / Declined** | `implementation` + `own-offline` |
