# Phase 89 Track A: FlashInfer NVFP4 MLA, Batched DeepGEMM FP8, GDN Train, PrimTS Keeps & 2-CTA MLA Prefetch, SM107 Split-K=8, and TileLang NVRTC

Deterministic Python 3.12 standard-library model reproducing the architectural invariants and performance ratios across the 10 Phase 89 Track A kernel/compiler commits (`flashinfer-ai/flashinfer` and `tile-ai/tilelang`).

## Usage

```bash
python experiments/flashinfer_nvfp4_mla_deepgemm_gdn_vsa_prims_sm107_tilelang_phase89/run.py
python experiments/flashinfer_nvfp4_mla_deepgemm_gdn_vsa_prims_sm107_tilelang_phase89/run.py --check
```
