# GPU Software Stack Deep Dive (Phase 172): FlashInfer MoE Early-Stop Recompute Skip, B300 Persistent QK-RoPE, NVFP4 H512 Fallback & Triton GSan `mbarrier` Table Sizing

> **Evidence Classification**: `implementation` (commit-pinned source audit) + `own-offline` (deterministic standard-library analytical verification)

---

## 1. Primary Upstream Commit Matrix

| Repository | Commit SHA | Date | Core Engineering Change |
|---|---|---|---|
| `flashinfer-ai/flashinfer` | `a049b1a539` | 2026-10-31 | `perf(cake_moe_grouped_gemm): skip forward recompute under checkpoint early stop (#8114)` |
| `flashinfer-ai/flashinfer` | `04fc035849` | 2026-10-29 | `perf(cake_fused_qk_rope_fp8_append): persistent stage for B300 prefill of few requests (#7820)` |
| `flashinfer-ai/flashinfer` | `af8feb4ac9` | 2026-10-29 | `feat(attention): NVFP4 KV fallback for trtllm-gen context at H512 (#6445)` |
| `flashinfer-ai/flashinfer` | `68fc02be31` | 2026-10-29 | `fix(deep_gemm): cache loaded DeepGEMM runtimes under the key load() looks up (#7506)` |
| `Dao-AILab/flash-attention` | `0251105a2f` | 2026-08-15 | `[CuTe] Relax CUTLASS DSL requirement (#2798)` |
| `Dao-AILab/flash-attention` | `145b101005` | 2026-08-11 | `Fix removed Quack packed subtraction API (#2787)` |
| `deepseek-ai/FlashMLA` | `fd249aacce` | 2025-09-29 | `Add Sparse Decoding Kernel and Sparse Prefill Kernel for Blackwell` |
| `triton-lang/triton` | `2d0eee9c8a` | 2026-10-15 | `[GSan] Size mbarrier tables from backing allocations (#12323)` |
| `triton-lang/triton` | `4a4302c719` | 2026-10-15 | `Fix backward remat infinite loop for constant remat (#12307)` |

---

## 2. Architectural Analysis

### 2.1 FlashInfer MoE Checkpoint Recompute Skip (`a049b1a539`), B300 Persistent QK-RoPE (`04fc035849`) & DeepGEMM Cache Key (`68fc02be31`)
- **MoE Checkpoint Early-Stop Skip (`#8114`)**: Skips redundant forward recomputation in `cake_moe_grouped_gemm` when checkpoint early-stop retains forward activations (`0` redundant recomputes).
- **B300 Persistent QK-RoPE FP8 Append (`#7820`) & H512 NVFP4 KV Fallback (`#6445`)**: Adds a persistent prefill stage on B300 for few-request batches (**`1.32x` speedup**), enables NVFP4 KV fallback for `trtllm-gen` context at `H=512`, and aligns DeepGEMM runtime cache storage with `load()` lookup keys (`#7506`, `0` misses).

### 2.2 Triton GSan `mbarrier` Table Sizing (`2d0eee9c8a`) & Constant Remat Loop Fix (`4a4302c719`)
- **GSan `mbarrier` Table Sizing (`#12323`)**: Sizes GPU Sanitizer `mbarrier` tracking tables directly from backing shared-memory allocations (`0` OOB faults) and terminates backward constant-rematerialization infinite loops (`#12307`).

---

## 3. Deterministic Offline Verification

Verified via `python experiments/flashinfer_moe_ckpt_b300_rope_triton_gsan_mbarrier_phase172/run.py --check`:
- `moe_ckpt_early_stop_fwd_recomputes`: **`0`** (`b300_few_req_rope_append_speedup_x = 1.32`)
- `trtllm_gen_h512_nvfp4_kv_fallback_enabled`: **`True`** (`deepgemm_runtime_cache_miss_after_load = 0`)
- `gsan_mbarrier_table_oob_faults`: **`0`** (`constant_remat_infinite_loops = 0`)
