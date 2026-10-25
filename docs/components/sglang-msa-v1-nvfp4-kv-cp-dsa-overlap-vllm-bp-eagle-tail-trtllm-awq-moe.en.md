# SGLang MSA-v1 NVFP4 KV Pool & CP DSA Indexer Overlap, vLLM Secondary-Tier Backpressure & EAGLE/MTP Partial-Tail Offload, and TensorRT-LLM MegaMoE NVFP4_AWQ Pre-Quant Scale

> **中文版本**: [sglang-msa-v1-nvfp4-kv-cp-dsa-overlap-vllm-bp-eagle-tail-trtllm-awq-moe.md](sglang-msa-v1-nvfp4-kv-cp-dsa-overlap-vllm-bp-eagle-tail-trtllm-awq-moe.md)  
> **Evidence ID**: `ev-sglang-msa-v1-nvfp4-kv-cp-dsa-overlap-vllm-bp-eagle-tail-trtllm-awq-moe-2026`  
> **Experiment**: [`experiments/sglang_msa_v1_nvfp4_kv_cp_dsa_overlap_vllm_bp_eagle_tail_trtllm_awq_moe/run.py`](../../experiments/sglang_msa_v1_nvfp4_kv_cp_dsa_overlap_vllm_bp_eagle_tail_trtllm_awq_moe/run.py)

---

## 1. First-Hand Source Audit Scope

This document audits seven first-hand commits across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM` (`#46141`, `#46139`, `#46140`, `#46179`, `#61780`, `#61019`, `#20823`):

1. **SGLang MSA-v1 Sparse Attention, NVFP4 KV Pool & MiniMax-M3.1 NVFP4 MoE (`#46141` `4991bd33ae`, `#46139` `174bd2976d`, `#46140` `a99c7d050f`)**: Implements native NVFP4 sparse KV pools (`3.56x` compression), `rcp.approx.ftz.f32` tie-exact reciprocal quantization, and 2-CTA PCIe `page_row_copy`.
2. **SGLang DSA Context-Parallel (CP) KV Gather Overlap with Indexer (`#46179`, `5971f47bc8`)**: Overlaps CP KV all-gather with `dsa_indexer` top-k selection via `run_indexer_with_kv_gather` in `interleave_bcg.py`.
3. **vLLM Tiered KV Offload Failure-Induced Backpressure (`#61780`, `2b1f75c80f`)**: Adds `on_store_failed()` to `EMABackpressureDetector` (`DEFAULT_FAILURE_THRESHOLD = 3`, `_FAILURE_EMA_FACTOR = 4.0`), halting `ENOSPC`/`EDQUOT` write storms on full secondary storage tiers.
4. **vLLM `OffloadingConnector` Partial-Tail Serving & Storing with EAGLE/MTP (`#61019`, `fc0cfcdc05`)**: Uses `eagle_proof_margin` to enable sub-block partial-tail KV offloading for EAGLE/MTP speculative decoding groups (`Qwen3-Next MTP`).
5. **TensorRT-LLM MegaMoE CuteDSL `NVFP4_AWQ` Pre-Quant Scale (`#20823`, `ecae3c97e2`)**: Applies `x = x * fc31_act_scale` across `fused_moe_cute_dsl.py`, `fused_moe_cute_dsl_b12x.py`, and `w4a16_nvfp4.py`.

---

## 2. Implementation Paths & Quantitative Analysis

### 2.1 SGLang MSA-v1 Sparse Backend & NVFP4 KV Pool (`#46141`, `#46139`, `#46140`)

- **NVFP4 KV Layout (`3.56x` Compression)**: In `nvfp4_store_kv_index.cuh`, every `kNvfp4BlockSize = 16` elements share one 1-byte `fp8_e4m3` scale plus `kNvfp4BlockBytes = 8` bytes of packed `e2m1` nibbles (`9 B` total vs `32 B` in BF16, achieving **3.56x** compression).
- **`rcp.approx.ftz.f32` Nibble Exactness**: The quantizer deliberately uses `rcp.approx.ftz.f32` rather than IEEE division because its `~1 ulp` difference crosses `e2m1` round-to-nearest ties on `~0.1%` of nibbles, matching `flashinfer::nvfp4_kv_quantize` bit-for-bit on B300/GB300.
- **2-CTA Host-Device Page Row Copy**: `kvcacheio/page_row.cuh` bounds host-device KV page transfers to `kPageRowBlockQuota = 2` CTAs (`kPageRowLanes = 64`), saturating PCIe while leaving **146 of 148 SMs (98.65%)** on B200 free for compute.

### 2.2 SGLang DSA CP KV Gather Overlap with Indexer (`#46179`)

In `interleave_bcg.py` and `forward_mla.py`, `run_indexer_with_kv_gather` pre-merges fused RoPE/FP8 Q/K tensors (`q.shape[-1] == kv_lora_rank + qk_rope_head_dim`) and overlaps CP KV all-gather (`185 us`) with `dsa_indexer` computation (`240 us`), reducing combined latency from `425.0 us` to `248.0 us` (**1.71x** speedup).

### 2.3 vLLM Secondary-Tier Store Failure Backpressure (`#61780`)

In `vllm/v1/kv_offload/tiering/backpressure.py`, `EMABackpressureDetector.on_store_failed()` counts consecutive failed stores (`success=False`). Reaching `DEFAULT_FAILURE_THRESHOLD = 3` sets the EMA to `_FAILURE_EMA_FACTOR = 4.0` times `high_water_mark`, causing `ThrottledDropPolicy(ramp_factor=2.0)` to drop **100%** of subsequent store attempts until idle decay allows a probe store.

### 2.4 vLLM `OffloadingConnector` Partial Tails with EAGLE/MTP (`#61019`)

In `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`, `supports_partial_tail` removes the exclusion of `is_eagle_group` by keying EAGLE partial tails at `eagle_proof_margin` past the boundary. For a 250-token prompt (`block_size=64`, `cache_hit_alignment_tokens=4`), prefix cache hit coverage increases from **75.0% (192/256)** to **96.88% (248/256)**.

### 2.5 TensorRT-LLM MegaMoE CuteDSL `NVFP4_AWQ` Pre-Quant Scale (`#20823`)

In `fused_moe_cute_dsl.py`, `fused_moe_cute_dsl_b12x.py`, and `w4a16_nvfp4.py`, AWQ/ARC checkpoints store `W / s` alongside `*.pre_quant_scale` (`fc31_act_scale`). Multiplying `x = x * fc31_act_scale` before FP4 quantization or W4A16 dequantized execution restores `(W / s) * (x * s) = W * x`, cutting relative L2 quantization error by **14.8x** (`0.2812 -> 0.0190`).

---

## 3. Summary Table

| Engine & Subsystem | Commit | Mechanism / Parameter | Baseline | Optimized | Improvement |
|---|---|---|---|---|---|
| SGLang MSA-v1 NVFP4 KV Pool | `#46141` / `#46139` | `kNvfp4BlockSize=16` (`9B` vs `32B` BF16) | $32\ \text{B / 16 elems}$ | $9\ \text{B / 16 elems}$ | **3.56x compression** |
| SGLang DSA CP KV Gather Overlap | `#46179` (`5971f47bc8`) | `run_indexer_with_kv_gather` (`CP=4`) | $425.0\ \mu\text{s}$ | $248.0\ \mu\text{s}$ | **1.71x speedup** |
| vLLM KV Offload Failure Backpressure | `#61780` (`2b1f75c80f`) | `DEFAULT_FAILURE_THRESHOLD=3`, `EMA=4.0x` | $0\%$ failure throttling | $100\%$ store drop | **Halts ENOSPC storm** |
| vLLM EAGLE/MTP Partial-Tail Offload | `#61019` (`fc0cfcdc05`) | `eagle_proof_margin`, `align=4` (`B=64`) | $75.0\%$ hit rate | $96.88\%$ hit rate | **+21.88 pts** |
| TRT-LLM MegaMoE `NVFP4_AWQ` Scale | `#20823` (`ecae3c97e2`) | `x = x * fc31_act_scale` | $\text{Rel-L2}=0.2812$ | $\text{Rel-L2}=0.0190$ | **14.8x lower error** |
