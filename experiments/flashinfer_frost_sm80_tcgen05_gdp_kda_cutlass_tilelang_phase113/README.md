# Phase 113 Track A: FlashInfer FROST SM80, `cake_tinygemm2_tcgen05`, GDP Preload, DeepGEMM `n_l2_group=8`, KDA & Blackwell Softmax + CUTLASS / TileLang

Deterministic Python 3.12 standard-library verification suite for Phase 113 (`priority: 122`) Track A:
- `flashinfer-ai/flashinfer@0e406f333b`: FROST low-latency grouped MoE SM80 (`sm_80` vs `sm_90a`/`sm_120a`) + typed activations
- `flashinfer-ai/flashinfer@8d9c839db0`: `cake_tinygemm2_tcgen05` 12 programs/arch (`36` total across `sm_100a/103a/107a`, `8.29 -> 7.26 us` `1.1419x` at `4x2880x2880`)
- `flashinfer-ai/flashinfer@9f6ba5a820` & `4d018bcd12`: `cake_gdp` dense-row preload (`1.07x` GB300) & `cake_batch_deepgemm_fp8` `K=7168, expected_m<=1` `n_l2_group=8` (`97,355 -> 0` bad elements)
- `flashinfer-ai/flashinfer@762b237e15` & `510e514b25`: `cake_kda` fused SM103 gate epilogue (`4 -> 3` tail launches) & `cake_blackwell_softmax` cached-cluster + persistent routes (`1.283x` B200 geomean across 41 shapes)
- `NVIDIA/cutlass@ad9bd53bda` & `a931725f1d` + `tile-ai/tilelang@d7a28bd78b` & `44405ee548`: CUTLASS per-group norm constant index capture (`37.5 -> 0.0` group-3 error) & TileLang Ascend PTO `vector_from_list` typed Ramp emission

```powershell
python experiments/flashinfer_frost_sm80_tcgen05_gdp_kda_cutlass_tilelang_phase113/run.py --check
```
