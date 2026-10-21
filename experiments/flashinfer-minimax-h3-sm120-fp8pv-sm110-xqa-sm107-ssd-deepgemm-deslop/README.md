# FlashInfer MiniMax-H3 SM120 FP8-PV Hybrid, SM110 XQA, SM107a Rubin SSD & DeepGEMM Deslop

Deterministic Python 3.12 standard-library reproduction of:
- `flashinfer-ai/flashinfer@ddba2f4b2c` (`#6936`): Opt-in `pv_mode="fp8"` (`NVFP4-QK / FP8-PV`) for MiniMax-H3 SM120 packed-varlen attention.
- `flashinfer-ai/flashinfer@da1e9f4bc2` (`#6945`): SM110 (`sm_110a`) XQA tree (`D512` `tmem`/`pair`) and decode (`D128`) vendorable `cake-build/1` package.
- `flashinfer-ai/flashinfer@7601d0bb7e` (`#6943`): Faster `sm_107a` (Rubin R200) look-back chunk-parallel Mamba2 SSD combined kernels.
- `flashinfer-ai/flashinfer@597e3a67ca` (`#6927`) & `4d79be4100` (`#6920`): DeepGEMM FP8 1D1D self-contained package and 5-family round-2 deslop (`-12,408` net lines).

## Run

```bash
python experiments/flashinfer-minimax-h3-sm120-fp8pv-sm110-xqa-sm107-ssd-deepgemm-deslop/run.py --check
```
