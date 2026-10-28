# Phase 121 Track A Experiment: FlashInfer CAKE Softmax Fence, MM FP4 Fallback, VC-Attention-QK8, MiniMax H3 SM103, Bounded Top-K, CUTLASS Block-Scale & TileLang

Reproducible Python standard-library model (`run.py`) and sealed output (`summary.json`) covering 10 upstream commits from `flashinfer-ai/flashinfer`, `NVIDIA/cutlass`, and `tile-ai/tilelang`:

- `flashinfer-ai/flashinfer@fc4df7e`: `cake_softmax` `fence.proxy.async.shared::cta;` before persistent softmax refills a freed tile slot (`4000` stress launches zero errors, `0.996x` 51-shape geomean on B200/GB300).
- `flashinfer-ai/flashinfer@dd21f60` & `015fff1`: `cake_mm_fp4` registered/fitting fallback plans (`unregistered_key` / `unfit_key`) and thread-safe `cake_qsa_nvfp4` plan-table eviction.
- `flashinfer-ai/flashinfer@e91141b` & `056670d`: `VC-attention-QK8` context FMHA (`1.35x` means / `1.44x` repair speedup on B200 Wan2.2 P1) and `cake_minimax_h3` SM103 (`sm_103a`) SMEM residual staging (`1.010x` geomean, `1.0138x` peak).
- `flashinfer-ai/flashinfer@93ab5d9`, `ba25ff7`, & `205c51e`: `delay_kernel` Torch stream binding (`5.147 ms` captured vs `0.019 ms`), bounded tensor `top_k_max` fast path (`3.7569x` B200 speedup at `vocab=262144, batch=256`), and radix top-k static SMEM accounting (`336 B` static SMEM on SM120, `558/558` pass).
- `NVIDIA/cutlass@ad9bd53` & `tile-ai/tilelang@0408712`: SM100/SM120 block-scale epilogue per-group `norm_constant` batch indexing (`l_norm_const`) and `without_kernel_cache()` test context manager.

## Usage

```bash
python experiments/flashinfer_cake_softmax_mm_fp4_vc_qk8_minimax_h3_sampling_cutlass_tilelang_phase121/run.py --check
```
