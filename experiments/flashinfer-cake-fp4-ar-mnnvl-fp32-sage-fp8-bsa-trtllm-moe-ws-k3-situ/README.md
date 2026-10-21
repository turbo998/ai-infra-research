# FlashInfer Cake NVFP4-Transport All-Reduce, MNNVL FP32 Square, Sage-FP8 BSA, TRT-LLM MoE Workspace & Kimi-K3 SiTU Folding

Deterministic Python 3.12 standard-library model reproducing:
- `flashinfer-ai/flashinfer@067b36956d` (#6954): Per-architecture (`sm_100a`/`sm_103a`/`sm_107a`) NVFP4-transport all-reduce + RMSNorm (`1.0718x` geomean, `1.0942x` peak at `8K x 6144`).
- `flashinfer-ai/flashinfer@1ff146212d` (#6818): MNNVL one-shot fused all-reduce + RMSNorm FP32 square fix (`|x| > 255.94` FP16 overflow prevention, Qwen3.5-27B RL validation `0.000 -> 0.698`).
- `flashinfer-ai/flashinfer@4e4c41da67` (#6872): SM100/SM103 Sage-FP8 block-sparse attention (`1.30x–1.40x`) and fused quantizers (`1.23x`).
- `flashinfer-ai/flashinfer@15c83f1c36` (#6825), `@75b30ffdbf` (#6898), & `@e2456a4eea` (#6879): `trtllm_fused_moe_workspace_size` (`3,579.9 -> 896.8 MiB`, `-2,683.1 MiB`), SM120 MegaMoE `part002.cuh` elimination, and Kimi-K3 SiTU `-17,018` lines (`75,311 -> 58,293`) guarded device header & binding folding.

## Run
```bash
python experiments/flashinfer-cake-fp4-ar-mnnvl-fp32-sage-fp8-bsa-trtllm-moe-ws-k3-situ/run.py --check
```
