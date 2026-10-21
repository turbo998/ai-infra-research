# FlashInfer SM90 Cake VSA 17-Program Dispatch, CUTLASS Fused-MoE PowLU Activation, Fused QK-RoPE Append & One-Wave DCP Decode

Deterministic Python 3.12 standard-library reproduction for Phase 52 (`priority: 61`) Track 1:

- `flashinfer-ai/flashinfer@ddf080d8a56c87f60a25ee9df750e373c4bcbe92` (`#6877`): SM90 `cake_vsa_sm90` self-contained 17-program BF16 backend (2 persistent + 15 small kernels), `119/137` benchmark rows faster on H100 SXM 80GB (`1.0515x` geomean on `ext33`, up to `1.2524x`).
- `flashinfer-ai/flashinfer@366214d4325e049199c9e79aa1eb30b0d3a867f1` (`#5743`): CUTLASS fused-MoE `PowLU` activation (`PowLU(g) = g > 0 ? g^(m / (sqrt(g) + 1)) * sigmoid(g) : g * sigmoid(g)`, `DEFAULT_POWLU_M = 2.5`, peak `4.02` at `m=2.5` and `7.02` at `m=3.5`), reusing `swiglu_alpha` and `swiglu_limit` with zero `ActivationParams` ABI change.
- `flashinfer-ai/flashinfer@51eb019768fc674a4feee33647bd389cc0bda1ce` (`#6888`) & `flashinfer-ai/flashinfer@49aa24fbc4b30a2ba0be61062342f891e8af2972` (`#6887`): Vendorable `cake_fused_qk_rope_fp8_append` and `cake_fused_qk_rope_append` across `sm_90a`/`sm_100a`/`sm_103a` (`1.1428x` BF16 geomean on GB300, `220/220` FP8 rows `>= 1.00x`).
- `flashinfer-ai/flashinfer@849e8be49ff27503ece16c27b54fcec2276ffdf8` (`#6122`) & `flashinfer-ai/flashinfer@90c82e1268c8d65e1f0476926cd6f8a0bdcbde4d` (`#6684`): Whole-tile static one-wave SM100a/SM103a DCP decode (`1.138x` BF16 `d=128`, `1.240x` FP8 `d=256` on GB300) and SM120 MegaMoE `part002.cuh` deduplication with 18 verified manifest seals.

## Run

```bash
python experiments/flashinfer-cake-vsa-sm90-powlu-moe-qk-rope-append-dcp-one-wave-megamoe/run.py --check
```
