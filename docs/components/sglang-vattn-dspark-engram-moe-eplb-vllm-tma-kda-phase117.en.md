# SGLang gfx950 MTP Full-CU Split, DSpark Hostless Draft Metadata, Engram Table Placement, EPLB Fused Recording & vLLM GLM-5.3 TMA MoE Acceleration (Phase 117 / Priority 126)

> **Evidence Classification**: Upstream `implementation` / `author-benchmark` (10 public upstream commits dated 2026-10-27) + repository `own-offline` deterministic scheduling & throughput modeling (`experiments/sglang_vattn_dspark_engram_moe_eplb_vllm_tma_kda_phase117/run.py`).

## 1. Architectural Problem Statement

Production serving runtimes for DeepSeek-V4 / V4.1 (DSpark / Engram / DSA), Kimi-K3 (KDA + MLA), and GLM-5.3 / Qwen3.5 addressed four critical runtime bottlenecks on 2026-10-27:

1. **CU Under-Utilization During Small-Batch MTP Verification on AMD MI350X (`gfx950`, 256 CUs)**: For small batches (`B=8, H_kv=4`), unsplit `vattn-asm` MTP verification only occupied `32` CUs (`12.5%` of `gfx950`).
2. **Host Sequence-Length Synchronization in DSpark Speculative Metadata**: Building DSpark draft block metadata from host-side sequence lengths introduced per-step D2H synchronization stalls during ragged speculative verification.
3. **EPLB Multi-Kernel Overhead in CUDA Decode Graphs & MegaMoE Local-vs-Global Token Heuristic Mismatch**: Separate expert-mapping and count-recording kernels inflated CUDA decode graph node counts, while passing rank-local token counts instead of global token counts to MegaMoE under-sized `M` tiles under Expert Parallelism (EP).
4. **GLM-5.3 MoE Up/Down Projection Memory Bottleneck & Kimi-K3 Unbudgeted KDA Workspace**: Enabling TMA in vLLM's fused MoE up/down projections unlocked `3.1%–3.4%` end-to-end throughput on GLM-5.3 (`author-benchmark`), while reserving Kimi-K3's KDA prefill workspace prior to memory profiling eliminated post-profile KV cache budget overruns.

---

## 2. Primary Upstream Implementation Evidence (10 Zero-Overlap Commits)

| # | Repository & Commit | Date | Runtime Path & Mechanism |
|---|---|---|---|
| 1 | `sgl-project/sglang@2c83dc6656` (`#45555`) | 2026-10-27 | `python/sglang/kernels/ops/attention/vattn_asm_gfx950/__init__.py`: Splits small-batch MTP verify attention across all `256` CUs on `gfx950` (`8.0x` active CU increase from `32` to `256` CUs). |
| 2 | `sgl-project/sglang@a0d4034d5b` (`#46891`) | 2026-10-27 | `python/sglang/srt/speculative/ragged_verify.py` & `deepseek_v4_backend.py`: Builds DSpark draft block metadata entirely on device without host sequence lengths (`68.0 us` saved per 4-step draft cycle). |
| 3 | `sgl-project/sglang@90a7339419` (`#46893`) | 2026-10-27 | `docs/docs/advanced_features/engram.mdx`: Adds `--engram-config` for DeepSeek-V4.1 to control Engram table placement across HBM and host DRAM (`18.0 GiB` HBM saved when offloading 75% of a 24 GiB table). |
| 4 | `sgl-project/sglang@f9f2665b6e` (`#44797`) | 2026-10-27 | `python/sglang/srt/layers/moe/topk.py`: Preserves `fp32` routing correction bias across both AITER and `radix4` dispatch paths, eliminating precision discontinuities across the `radix4` batch cutoff. |
| 5 | `sgl-project/sglang@7e142dd9fd` (`#46093`) | 2026-10-27 | `python/sglang/srt/layers/moe/mega_moe.py`: Always passes the global token count to the Mega MoE tile heuristic so EP ranks select optimal `M=64` tiles instead of `M=16`. |
| 6 | `sgl-project/sglang@baa349af7b` (`#46497`) | 2026-10-27 | `python/sglang/kernels/ops/moe/eplb_map_record.py`: Fuses EPLB expert-count recording into a single kernel inside CUDA decode graphs (`3 -> 1` kernel nodes). |
| 7 | `sgl-project/sglang@3145c7ae66` (`#43839`) | 2026-10-27 | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`: Online-quantizes Qwen3.5 BF16 GDN projections to `MXFP4` (`4.25` effective bits/weight, `3.7647x` weight compression). |
| 8 | `sgl-project/sglang@7f07519943` (`#43271`) | 2026-10-27 | `python/sglang/srt/layers/attention/linear/kda_backend.py` & `dsa_indexer_kpool.py`: Masks MoE padding tokens during hybrid DSA/KDA MTP verification (`0` state corruption tokens). |
| 9 | `vllm-project/vllm@c56706e88e` (`#63820`) | 2026-10-27 | `vllm/model_executor/layers/fused_moe/fused_moe.py`: Enables TMA for MoE up/down projections, achieving **3.1%–3.4%** E2E throughput gain on GLM-5.3 (`author-benchmark`, modeled at `3.3%`). |
| 10 | `vllm-project/vllm@e96b490b12` (`#63989`) | 2026-10-27 | `vllm/models/kimi_k3/nvidia/kda.py`: Reserves Kimi-K3 KDA prefill workspace (`384 MiB`) before memory profiling (`0.0 MiB` unbudgeted KV cache overrun). |

---

## 3. Deterministic Verification Summary (`run.py --check`)

- `gfx950_cu_utilization_gain_ratio`: `8.0` (`32 -> 256` active CUs during small-batch MTP verify)
- `dspark_step_overhead_saving_us`: `68.0` (`4` draft steps without host `seq_lens` synchronization)
- `eplb_decode_graph_node_reduction`: `2` (`3 -> 1` kernel in CUDA decode graph)
- `gdn_mxfp4_weight_compression_ratio`: `3.7647` (`16.0 / 4.25` effective bits per weight)
- `vllm_glm53_tma_e2e_gain_pct`: `3.3` (within upstream-reported `3.1%–3.4%` E2E improvement)
- `reserved_kv_budget_overrun_mib`: `0.0` (pre-profile KDA workspace reservation eliminates OOM drift)
