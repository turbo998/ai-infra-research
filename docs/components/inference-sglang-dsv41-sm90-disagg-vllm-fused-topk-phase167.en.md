# Inference & Sparse Indexing Phase 167: SGLang dsv41 SM90 Paged Decode Tile Balancing, Disaggregation First-Refusal Stop, and vLLM Single-Launch Fused Top-K Prefill

## 1. Architectural Context & Problem Statement

Production disaggregated serving and sparse-attention indexing face three acute tail-latency and memory-bandwidth bottlenecks:
1. **Straggler SM Imbalance in dsv41 SM90 Paged Decode Indexer**: When a decode batch mixes many short contexts (e.g., fifteen `4K` sequences) with a single long context (`128K`), sequence-level CTA scheduling pins all `128` tiles of the long request onto a single SM wave while leaving other SMs idle.
2. **Redundant Disaggregation Admission Probing & Mamba HiCache Slot Consumption**: Continuing to probe queued decode transfers in a disaggregation pass after the first memory admission refusal wastes scheduler CPU cycles; meanwhile, storing Mamba recurrent state checkpoints into HiCache previously consumed an active GPU KV slot.
3. **Logits Materialization in vLLM Sparse-Indexer Prefill**: Writing the full `[T_prefill, N_candidates]` FP32 score matrix to HBM before launching a separate top-k selection kernel creates a severe HBM bandwidth bottleneck during long-context prefill.

---

## 2. Primary Upstream Mechanism Deep Dive (10 Zero-Overlap Commits)

### 2.1 SGLang: SM90 dsv41 Tile Balancing, First-Refusal Stop, AMD Mamba Slotless HiCache & GB300 DFLASH
- **`2ccbdab65f` (`2026-10-31`, `#47387`)**: Balances SM90 paged decode indexer tiles (`fp4_indexer.py`) evenly across all `132` GPU SMs so a single long-context request no longer dictates the step latency.
- **`566dbd2cb3` (`2026-10-31`, `#46898`)**: Halts device admission in a disaggregated decode transfer pass (`decode.py`) immediately at the first refusal.
- **`2f6e7f3475` (`2026-10-31`, `#44718`)**: Enables AMD Mamba models to store HiCache checkpoints without allocating a GPU slot.
- **`b20fdd2eb3` (`2026-10-31`, `#45876`)**: Routes DFLASH draft BF16 GEMMs through tuned CuTeDSL dispatch (`cutedsl_bf16_gemm.py`) on GB300.

### 2.2 vLLM: Single-Launch Fused Scoring + Top-K Prefill Kernel & XPU Release Index
- **`25fc60723c` (`2026-10-31`, `#60618`)**: Introduces a single-launch fused scoring and top-k CUDA kernel (`csrc/fused_mqa_topk/`) for sparse-indexer prefill, computing scores and selecting top-k candidates in registers/shared memory without materializing intermediate logits in HBM.
- **`00ba900aa8` (`2026-10-30`, `#64633`)**: Uses the GitHub release page index URL for `vllm-xpu-kernels`.

### 2.3 TensorRT-LLM & DeepEP: GLM4-MoE Coverage, Build Retry & Node 24 CI
- **`04f92470a5` (`2026-10-30`, `#21256`) & `3a20b06247` (`2026-10-30`, `#21127`)**: Adds `Glm4MoeForCausalLM` test coverage and build-stage retry resilience in TensorRT-LLM.
- **`3fcf25ed1b` (`2025-12-24`, `#528`) & `b57e5e212a` (`2025-12-05`, `#512`)**: Upgrades DeepEP GitHub Actions for Node 24 and documents experimental optimization branches.

---

## 3. Deterministic Offline Verification Summary

Validated via `experiments/sglang_dsv41_sm90_disagg_stop_vllm_fused_topk_phase167/run.py`:

| Subsystem | Baseline Metric | Phase 167 Optimized Metric | Architectural Impact |
|---|---|---|---|
| SGLang dsv41 SM90 Paged Decode (`15×4K + 1×128K`, `132` SMs) | `128` max tiles on 1 SM | `2` max tiles per SM | **`64.0x` tail-step tile-wave reduction** |
| SGLang Disagg Admission & Mamba HiCache | `32` probes; `1` GPU slot | `4` probes (first refusal); **`0` GPU slots** | **`87.5%` fewer admission probes** |
| vLLM Fused Sparse-Indexer Prefill (`T=8192, N=4096, K=64`) | `128 MiB` logits materialized | `2 MiB` top-64 indices written | **`64.0x` HBM traffic reduction** |
