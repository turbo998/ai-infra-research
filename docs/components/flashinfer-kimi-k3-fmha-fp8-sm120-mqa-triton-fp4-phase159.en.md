# FlashInfer Kimi-K3 Dense Route Retuning, One-Stage FP8 Paged Context, SM120 CTA-Parallel MQA & Triton Native FP4 / gfx1250 TDM Scatter Deep Dive (Phase 159)

> **Evidence Mode**: `implementation` (fixed upstream commit source audit) + `own-offline` (deterministic Python 3.12 standard-library model in `experiments/flashinfer_kimi_k3_fmha_fp8_sm120_mqa_triton_fp4_phase159/run.py`). No GPU hardware benchmark executed.

## 1. Primary Source Matrix (10 Zero-Overlap Upstream Commits)

| Source ID | Repository | Commit (`2026-10`) | Key Technical Path & Implementation Change |
|---|---|---|---|
| `flashinfer-cake-kimi-k3-attn-res-b200-b300-gb300-2026` | `flashinfer-ai/flashinfer` | `6bb09734fe` (`#8120`) | Retunes dense routes in `cake_kimi_k3_attn_res` across B200, B300, and GB300 SKUs. |
| `flashinfer-cake-fmha-one-stage-fp8-paged-context-2026` | `flashinfer-ai/flashinfer` | `7aef24a430` (`#8101`) | Adds multi-wave and short-KV one-stage FP8 paged context programs in `cake_fmha`. |
| `flashinfer-cake-deepgemm-sm120-fp8-mqa-logits-cta-2026` | `flashinfer-ai/flashinfer` | `4d601a17ad` (`#8099`) | Adds SM120 paged FP8 MQA logits CTA-parallel scheduler routes for RTX 5090 and RTX PRO 6000. |
| `flashinfer-cake-moe-grouped-gemm-nondet-training-2026` | `flashinfer-ai/flashinfer` | `e1b2a15998` (`#8088`) | Introduces opt-in non-deterministic MoE grouped GEMM training package. |
| `megatron-mcore-determinism-debug-skill-2026` | `NVIDIA/Megatron-LM` | `902feb8eea` (`#7262`) | Adds `mcore-determinism-debug` workflow for bisecting bitwise drift across TP/PP/EP ranks. |
| `megatron-training-validation-config-migrate-2026` | `NVIDIA/Megatron-LM` | `1bb190d467` (`#8149`) | Migrates and decouples `TrainingConfig` and `ValidationConfig` contracts. |
| `triton-amd-gfx1250-multi-cta-tdm-scatter-2026` | `triton-lang/triton` | `6bf64a51df` (`#12324`) | Enables Multi-CTA Tensor Data Mover (TDM) Scatter on AMD `gfx1250`. |
| `triton-native-fp4-e8m0-types-support-2026` | `triton-lang/triton` | `4283dc7bb5` (`#12315`) | Adds first-class native `fp4` and `e8m0` data types in Triton IR and frontend. |
| `tilelang-atomic-reduction-dest-element-2026` | `tile-ai/tilelang` | `fdb19f26d6` (`#3340`) | Explicitly names the destination element in TileLang atomic reduction examples. |
| `tilelang-bisheng-cleanup-temp-files-2026` | `tile-ai/tilelang` | `18b439c888` (`#3461`) | Honors `TILELANG_CLEANUP_TEMP_FILES` in the Ascend `bisheng` wrapper (`0.0 MB` temp leak). |

## 2. Quantitative Analytical Summary (`own-offline`)
- **B200 / B300 / GB300 Kimi-K3 Dense Routes**: Retuned routes deliver **$1.18\times$–$1.20\times$** throughput gains ($1510.0 \to 1812.0\text{ TFLOPS}$ on GB300).
- **One-Stage FP8 Paged Context & SM120 MQA Logits**: Cuts short-KV ($L_{kv}=256$) latency by **$32.93\%$** ($41.0 \to 27.5\,\mu\text{s}$) and speeds up SM120 4K MQA logits by **$1.6364\times$** ($198.0 \to 121.0\,\mu\text{s}$).
- **MoE Grouped GEMM & Triton `gfx1250` / `fp4`**: Opt-in non-deterministic MoE grouped GEMM achieves **$1.3718\times$** speedup at 256 experts; 8-CTA TDM Scatter achieves **$4.1875\times$** scatter bandwidth ($2680\text{ GB/s}$).
