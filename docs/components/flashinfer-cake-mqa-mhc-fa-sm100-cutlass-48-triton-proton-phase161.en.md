# FlashInfer CAKE DeepGEMM Dense MQA / mHC Autograd, FlashAttention SM100 Sparse MLA, CUTLASS 4.8 & Triton Proton Trace Isolation Deep Dive (Phase 161)

> **Evidence Classification**: Primary upstream commit analysis (`implementation`: `fe08dc597e95`, `9f6a84c6ae80`, `deacae9249c2`, `dd8325c1f547`, `843bf0b86bda`, `616b0e8abab1`, `784aee1878bd`, `0b55a2f691d6`, `098de2a652cf`, `cdcf8d86daa9`, `2bdead3a37e9`, `f53372790560`) + deterministic Python 3.12 standard-library model (`own-offline`: `experiments/flashinfer_cake_mqa_mhc_fa_sm100_cutlass_48_triton_proton_phase161/run.py`). Not a live GPU hardware benchmark.

## 1. Core Architectural Findings & Implementation Paths

1. **FlashInfer CAKE DeepGEMM Dense MQA Single-Launch Chain Extension (`fe08dc597e95`)**:
   - Extends the single-launch `FP8 Q = 1 / Q = 128` and `FP4 Q = 1` chains by `+1` KV block (`7 -> 8` blocks), avoiding a split kernel launch and secondary LSE merge at the 8-block boundary (`6.80 us -> 3.40 us`, `-50.0%` launch overhead in `own-offline` modeling).
2. **FlashInfer CAKE mHC Opt-In Non-Deterministic Autograd (`deacae9249c2`), Single-Thread Compacted Logprob (`9f6a84c6ae80`) & Deterministic KCP Causal Conv1D Names (`dd8325c1f547`)**:
   - `deacae9249c2` introduces an opt-in non-deterministic backward backend for mHC (`148.0 us -> 112.48 us`, `24.0%` faster in `own-offline` modeling), while `9f6a84c6ae80` validates compacted LM-head logprob statistics in 1 CPU thread and `dd8325c1f547` standardizes 4 KCP context-parallel causal conv1d entry points.
3. **FlashAttention SM100 Sparse MLA Exact Max (`784aee1878bd`) / TVM-FFI 0.1.11 (`843bf0b86bda`) / CUDA 13 Windows Build (`616b0e8abab1`), CUTLASS 4.8 (`0b55a2f691d6`, `098de2a652cf`, `cdcf8d86daa9`) & Triton Proton Isolation (`2bdead3a37e9`, `f53372790560`)**:
   - Preserves SM100 Sparse MLA exact max invariants (`0.0 ULP` drift), isolates `ChromeTraceWriter` artifacts (`0.0%` file collision rate), and migrates Triton MLIR builders away from deprecated `Op::create`.

## 2. Deterministic Verification

```bash
python experiments/flashinfer_cake_mqa_mhc_fa_sm100_cutlass_48_triton_proton_phase161/run.py --check
```
