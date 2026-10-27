# Phase 103 Track A: FlashInfer Blackwell BF16xFP4 & W4A16 MegaMoE Deterministic Model

Deterministic Python 3.12 standard-library model reproducing:
- `flashinfer-ai/flashinfer@535c18c714` (#7463): `cake_blackwell_bf16_fp4` relaxed `tcgen05` cluster entry arrive & 8-CTA split-K DSMEM reduce-scatter payload reduction (`7,168 B -> 448 B`, `16.0x`).
- `flashinfer-ai/flashinfer@124955a41b` (#7557): `sm103_bf16_nvfp4_bf16_cake` W4A16 MegaMoE EP backend (`1.5912x` at `T=1`, `1.0839x` at EP8 `T=4096`).
- `flashinfer-ai/flashinfer@cbe168b01e` (#5196): `SamplingFromLogitsKernel` deferred logits block reduction (`38 -> 1` reductions at `V=151,936`).
- `NVIDIA/cutlass@6278db36ee` (#3725), `af3d4367a8` (#3639) & `tile-ai/tilelang@a765bb4bf2` (#3356), `5434f0ac54` (#3346).

## Usage
```bash
python experiments/flashinfer_blackwell_bf16_fp4_w4a16_megamoe_phase103/run.py
python experiments/flashinfer_blackwell_bf16_fp4_w4a16_megamoe_phase103/run.py --check
```
