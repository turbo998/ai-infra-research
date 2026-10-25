# Deep Dive into TensorRT-LLM DeepSeek-V4 Rubin SM107 & CuteDSL MoE, vLLM MLA/Adaptive Speculative Sampling, SGLang HiCache, and AI Infra Innovators (Fireworks/Together/Cerebras/Groq)

> **中文版**: [trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache.md](trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache.md)  
> **Phase**: 72 (`priority: 81` · Track B) · **Verified Date**: `2026-10-25` (UTC+08:00)  
> **Verification Script & Sealed Summary**: [`experiments/trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache/run.py`](../../experiments/trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache/run.py) · [`summary.json`](../../experiments/trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache/summary.json)

---

## 1. Architectural Context and Core Problems

Production inference stacks serving frontier sparse MoE and multimodal models (DeepSeek-V4 / DSpark, GLM 5.3, Gemma4) across TensorRT-LLM, vLLM, and SGLang addressed five critical bottlenecks on `2026-10-24`–`2026-10-25`:

1. **Redundant HBM Traffic in DeepSeek-V4 MLA Projections on Rubin (`sm_107`)**: Unfused paths materialize BF16 `q_pe` and placeholder Q buffers before RoPE/FP8 quantization and allocate intermediate `o_b_proj` buffers before copying into the residual output.
2. **Host-Stream Overhead in Async MoE Input Quantization and Missing `NVFP4_AWQ` Pre-Quant Scales**: Overlapping CuteDSL NVFP4 input quantization on an auxiliary stream saves `~25 us/layer` at `batch=8` on GB300, but in eager single-stream runs the host fork/join overhead dominates. Meanwhile, dropping `pre_quant_scale` (`fc31_act_scale`) when streaming `NVFP4_AWQ`/`NVFP4_ARC` experts quantizes unsmoothed activations against smoothed calibration scales.
3. **Shape Disagreement Between CPU Batch-Sharded Sampling and GPU Adaptive Speculative Verification in vLLM**: `BatchSharder` builds its TP all-to-all plan on the CPU from scheduled draft upper bounds, whereas `AdaptiveVerificationManager` dynamically trims admitted draft counts on the GPU.
4. **EAGLE Bigram Boundary Truncation in SGLang PD Disaggregated HiCache Restores**: An EAGLE tree indexes $N$ bigrams using $N+1$ token IDs; matching only `decode_prefix_len` drops the trailing boundary token during decode L2/L3 cache restores.
5. **AI Infra Innovator Architecture Trade-Offs (Fireworks / Together vs. Cerebras / Groq)**: Serving 400B–800B+ sparse MoE models (`42B` active parameters) exposes fundamental capacity and bandwidth boundaries between wafer/on-die SRAM engines (Cerebras WSE-3, Groq LPU) and HBM3e/HBM4 + NVFP4/MXFP8 disaggregated speculative clouds (Fireworks AI, Together AI).

---

## 2. First-Hand Upstream Code Audit and Mechanics

### 2.1 TensorRT-LLM `d7e0ecac925d` (`#19961`), `032eb456ecb1` (`#20808`), `ecae3c97e2b6` (`#20823`), `abb94ab8fc1c` (`#20663`) & `9666d7427a8c` (`#20774`)

1. **DeepSeek-V4 / DSpark Optimizations on Rubin SM107 (`#19961`)**:
   - **Default CuteDSL Block-Scaled GEMM on SM107**: Sets `"use_cute_dsl_blockscaling_mm": get_sm_version() == 107` in `modeling_deepseekv4.py`.
   - **Fused Rotated FP8 Q (`q_rope_applied = True`)**: In `attentionOp.cpp`, `mla_params.q_rope_applied = mla_params.fuse_q_fp8_in_rope && !q_pe.has_value()` lets the upstream `q_b` GEMM fusion write the complete rotated FP8 Q directly into `quant_q_buffer` without a separate `q_pe` tensor or redundant RoPE rewrite, while pre-registering `_quant_scale_qkv` to avoid H2D copies during first CUDA Graph capture.
   - **In-Place SM107 MXFP8 `o_b_proj` (`allow_unpartitioned_output = True`)**: `_rubin_dsv4_o_b_proj_out` writes the MXFP8 GEMM output directly into `output` in place, eliminating **`79,872 B/token`** of intermediate HBM traffic (`128` heads, `q_dim=192`, `hidden=7168`).
   - **CuteDSL NVFP4 MegaMoE Acceptance Length Baseline**: Records `ref_al = 4.433448` (`min_al = 4.211776`, `96.21%` GSM8K on GB300) for `TestDeepSeekV4ProDSpark::test_gsm8k_dep4_megamoe_cutedsl`.
2. **Multi-Stream Gating for CuteDSL MoE Async Input Quantization (`#20808`)**:
   - `_should_quantize_input_async` forks `quantize_input_async` onto the backend auxiliary stream **only** when `do_multi_stream()` or `torch.cuda.is_current_stream_capturing()` holds (and `comm is None` and `not apply_router_weight_on_input`), saving `~25 us/layer` (`1.525 ms` across 61 layers at `batch=8` on GB300) without penalizing host-bound eager forwards.
3. **`NVFP4_AWQ` / `NVFP4_ARC` Pre-Quant Scale Application (`#20823`)**:
   - Stages `w1_pre_quant_scale` / `w3_pre_quant_scale` during streaming NVFP4 expert loads and multiplies `x = x * self.fc31_act_scale` across all six CuteDSL/CUTLASS NVFP4 and W4A16 MoE backends before `fp4_quantize`.
4. **Gemma4 Multimodal Encoder Budget (`#20663`) & Per-Request LoRA Format Reload (`#20774`)**:
   - Computes `_padded_patches_per_frame = max_soft_tokens * pooling_kernel_size^2` (`2520` patches/frame for `280` soft tokens and `3x3` pooling) and audio `4x` subsampling so multimodal items obey `max_num_encoder_tokens`.
   - Propagates `py_lora_ckpt_source` on `LlmRequest` so `PeftCacheManager.add_request_peft` reloads uncached adapters using the request's validated checkpoint format (`"hf"` vs `"nemo"`).

---

### 2.2 vLLM `56e4e42868b4` (`#62102`), `7050ef881ff0` (`#58466`) & `5f63ea461ccd` (`#63001`)

1. **Compile-Time `skip_scale` in `concat_and_cache_mla` (`#62102`)**:
   - Specializes `concat_and_cache_mla_kernel<..., bool skip_scale>` so unit FP8 scale (`_k_scale_float == 1.0`) bypasses device pointer dereference and scale arithmetic, delivering a **1.2%–3.0% E2E TPOT improvement** on GLM 5.3 / DeepSeek-V3.2.
2. **Batch-Sharded Sampling with GPU Adaptive Verification (`#58466`)**:
   - Introduces `AdmittedDraftCounts` (pinned CPU buffer + CUDA event) in `AdaptiveVerificationManager` so `BatchSharder.shard_sampler_inputs` sizes its TP all-to-all gather from the actual GPU-admitted draft counts (`1 + num_admitted_draft_tokens_np`) rather than upper-bound scheduled counts.
3. **ModelOpt Mixed-Precision DFlash/DSpark Draft Layer Mapping (`#63001`)**:
   - Maps `midlayer -> layers.0` and offsets `layers.{i} -> layers.{start_layer_id + i}` in descending layer order (`reversed(range(num_hidden_layers))`) to prevent cascading renames (`0 -> 1 -> 2`), while normalizing `E4M3FN -> E4M3FNUZ` on ROCm.

---

### 2.3 SGLang `0ff67ee72aaf` (`#45076`), `f82af9ca17c9` (`#40595`) & `8ef5dec95faa` (`#46047`)

1. **EAGLE Bigram Boundary Token in PD Decode HiCache Restores (`#45076`)**: Adds `_bigram_boundary_tokens(tree_cache) = int(tree_cache.tree_core.is_eagle)` to `pm.decode_prefix_len` during `match_kv_cache` in `_try_hicache_queue_load_back`.
2. **Out-of-Tree Dynamic External Linkers for `UnifiedRadixCache` (`#40595`)**: Adds `--unified-cache-external-linker-backend=dynamic` and `--unified-cache-external-linker-config` to link `UnifiedRadixCache` directly to external KV stores without a host DRAM tier.
3. **Eliminating Per-Step Host Syncs in Video Diffusion (`#46047`)**: Memoizes RoPE tables and conditioning masks per chunk and replaces `list[int]` indexing in `cosmos_dreams` with `list[slice]`, reducing per-step H2D syncs to `0`.

---

## 3. AI Infra Innovators Comparison: Fireworks & Together vs. Cerebras & Groq

| Dimension | Fireworks AI / Together AI (Disaggregated GPU Cloud + SpecDecode) | Cerebras WSE-3 (Wafer-Scale SRAM) | Groq LPU (Deterministic SRAM TSP) |
|---|---|---|---|
| **Memory Hierarchy** | `288 GB` HBM3e (`8 TB/s` on GB300) / HBM4 on Rubin + NVFP4/MXFP8 | `44 GB` on-wafer SRAM (`21 PB/s`) + off-wafer MemoryX | `230 MB` on-die SRAM per LPU (`80 TB/s`) |
| **816B-A42B / DeepSeek-V4 Fit** | **Native fit**: `~408 GB` NVFP4 total weights fit in 2 GB300 GPUs; only `21 GB` active bytes read per step | **Hybrid**: Total weights exceed `44 GB` wafer SRAM; requires MemoryX streaming or multi-wafer pipeline | **Capacity-bound**: Requires >1,800 LPUs just to hold 408 GB of weights in SRAM |
| **Effective Decode Speed** | `~2.63 ms` ideal active-weight read on 1 GB300; with DSpark (`AL=4.43`), reaches **~1,200+ tok/s** per stream at high batch efficiency | **1,500–2,000+ tok/s** at `batch=1` when weights fit on-wafer | **300–800+ tok/s** with near-zero jitter for dense models `<=70B` |

---

## 4. Deterministic Verification Experiment

```bash
python experiments/trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache/run.py --check
```
