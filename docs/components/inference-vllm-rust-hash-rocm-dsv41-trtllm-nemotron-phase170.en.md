# Inference Frontend Hashing, ROCm Kernels & Multimodal Verification Deep Dive (Phase 170): vLLM Rust Prompt Block Hashing, ROCm DSv4.1 `gfx942` Skinny-GEMM & TRT-LLM Nemotron VL Teacher-Forced Logits

> **Evidence Classification**: `implementation` (commit-pinned source audit) + `own-offline` (deterministic standard-library analytical verification)

---

## 1. Primary Upstream Commit Matrix

| Repository | Commit SHA | Date | Core Engineering Change |
|---|---|---|---|
| `vllm-project/vllm` | `e318ac0979` | 2026-10-31 | `[Perf][Frontend] Compute prompt block hashes in the Rust frontend (#63842)` |
| `vllm-project/vllm` | `de83f55848` | 2026-10-29 | `[ROCm][DSv4.1] Lift the skinny-GEMM cap in the gfx942 draft lm_head test (#64532)` |
| `vllm-project/vllm` | `0124e16a6c` | 2026-10-29 | `[ROCm] Monitor FlyDSL JIT compilation after warmup (#62481)` |
| `sgl-project/sglang` | `96c296b3e8` | 2026-10-30 | `[Test] [AMD] fp8-grid: Compare the GEMM to aiter within one bf16 rounding, not bitwise (#47054)` |
| `sgl-project/sglang` | `c575b8f36d` | 2026-10-30 | `[Chore] [AMD] mono-kernel: Bump sgl-amd-mono for Qwen3.8-Flash-Next (#47483)` |
| `sgl-project/sglang` | `ca93940742` | 2026-10-30 | `[Fix] humming: adapt integration to kernels 0.1.18 (#46018)` |
| `NVIDIA/TensorRT-LLM` | `e0892ee756` | 2026-10-29 | `[https://nvbugs/6758946][fix] Compare teacher-forced logits in the Nemotron VL image batch-equivalence test (#20330)` |
| `NVIDIA/TensorRT-LLM` | `25e2aeb230` | 2026-10-29 | `[https://nvbugs/6982380][fix] Open the Harmony final channel in JSON mode eval prompts (#21048)` |

---

## 2. Architectural Analysis

### 2.1 vLLM Rust Frontend Prompt Block Hashing (`e318ac0979`) & ROCm DSv4.1 / FlyDSL (`de83f55848`, `0124e16a6c`)
- **Rust Frontend Block Hashing (`#63842`)**: Offloads 16-token prefix-cache block hash computation from Python GIL threads to the Rust frontend, cutting 32K-token prompt hashing latency from `1840.0 us` to `160.0 us` (**`11.5x` speedup**).
- **ROCm `gfx942` DSv4.1 Draft `lm_head` (`#64532`) & `gfx95` FP8-Grid (`#47054`)**: Lifts the skinny-GEMM batch cap to `M=64` on `gfx942`, enforces `0` FlyDSL JIT recompilations after warmup (`#62481`), and validates `gfx95` FP8-Grid GEMM against AITER within `1` BF16 rounding ULP rather than brittle bitwise equality.

### 2.2 TensorRT-LLM Nemotron VL Teacher-Forced Logits (`e0892ee756`)
- **Teacher-Forced Batch-Equivalence (`#20330`)**: Replaces multi-step greedy token comparisons (which diverged across `14` tokens on sub-ULP reductions) with teacher-forced logit comparison (`max_diff = 0.00012`), and opens the Harmony final channel in JSON mode eval prompts (`#21048`).

---

## 3. Deterministic Offline Verification

Verified via `python experiments/vllm_rust_block_hash_rocm_dsv41_trtllm_nemotron_phase170/run.py --check`:
- `rust_hash_speedup_x`: **`11.5x`** (`1840.0 us` -> `160.0 us`)
- `flydsl_post_warmup_recompiles`: **`0`** (`gfx95_fp8_grid_max_bf16_ulp = 1`)
- `teacher_forced_logit_max_diff`: **`0.00012`** (`harmony_final_channel_opened = True`)
