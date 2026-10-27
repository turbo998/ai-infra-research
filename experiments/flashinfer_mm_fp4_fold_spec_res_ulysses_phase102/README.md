# Phase 102 Track A: FlashInfer `cake_mm_fp4` Cluster-Split & Token-Remainder Folds, Deferred Speculative Sampling & Cake Ulysses Direct A2A

Deterministic Python 3.12 standard-library model reproducing:
- **FlashInfer `cake_mm_fp4` (`cb138f8aa6`, #7550)**: Standalone per-token NVFP4 quantizer 2/4/8-CTA cluster splits (`1.027x–1.086x` on B200/GB300) and token-remainder GEMM folds for `M=129..136` and `M=257..272` (`1.233x` on B200, `1.194x` on GB300).
- **FlashInfer `ChainSpeculativeSampling` (`03bd5b3402`, #5201)**: Thread-local accumulation across 32 vocabulary tiles (`V=128,256`), replacing 32 `BlockReduce` + `__syncthreads()` barriers with 1 (`1.66x` peak on A100, `1.65x` on H100).
- **FlashInfer Cake Ulysses World-2 Direct A2A (`88205170de`, #7566)**: Concurrent 128 peer-stream CTAs + 20 self-copy CTAs and unconditional symmetric handshake (`1.039x` on B200, `1.030x` on B300).
- **CUTLASS CuTeDSL & TileLang (`c8fc8a6535`, `218d3bfc4a`, `837486abf7`, `0d75cd9af5`)**: `local_tile` output shape vs `zipped_divide` rest mode, `tma_gather4`/`tma_scatter4` `int32` constant row normalization, and tensor-device stream resolution.

## Usage
```bash
python experiments/flashinfer_mm_fp4_fold_spec_res_ulysses_phase102/run.py
python experiments/flashinfer_mm_fp4_fold_spec_res_ulysses_phase102/run.py --check
```
