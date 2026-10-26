# SGLang DSV4/V4.1 Prefill Graph Bridges & Candidate Tables, Kimi-K3 TGV BF16 GEMMs, Wan Direct Ulysses Blocks, vLLM MRV2 Input Release & DecodeBenchConnector (Phase 96)

> **Evidence Classification**: `implementation` (pinned upstream commits) + `own-offline` (deterministic Python 3.12 standard-library verification; not GPU hardware profiling).

## 1. Primary Upstream Source Evidence (10 Zero-Overlap Commits)

| Source ID | Repo / Commit | Key Implementation Path & Architectural Mechanism |
|---|---|---|
| `sglang-kernels-beam-model-support-pr45254-2026` | `sgl-project/sglang` `66f79f1a90` (PR #45254) | Adds `BeamForCausalLM` fused MoE Triton configs and rounds DeepSeek-V4 SWA K/V to BF16 before FP8 quantization to match QAT and DSA prefill-CP. |
| `sglang-dsv4-replay-tail-caller-prefill-graph-bridges-pr46516-2026` | `sgl-project/sglang` `3d2a781b3e` (PR #46516) | Writes DeepSeek-V4 replay-tail outputs via `_write_bridge` into caller-held `_bridge_buffers`, keeping older smaller buffer sets alive for already-captured CUDA graphs. |
| `sglang-kimi-k3-tgv-bf16-gemm-narrow-mla-gate-pr46468-2026` | `sgl-project/sglang` `a0618d5e65` (PR #46468) | Routes Kimi-K3 DSpark draft `(7168, 1792)`, `(7168, 512)` and dense MLP `(8448, 7168)` BF16 GEMMs to CuTeDSL TGV and narrows `mla_output_gate_kernel` from `kVecN = 8` to `kVecN = 4` (`8 B`). |
| `sglang-dsv41-candidate-table-caller-prefill-graph-buffers-pr46399-2026` | `sgl-project/sglang` `e8019014b0` (PR #46399) | Writes DeepSeek-V4.1 sparse indexer candidate tables into caller-held `_graph_tables` (newest first, preserving older tables for captured graphs) and removes `is_capturing()`. |
| `sglang-dsv41-align-bf16-prefill-indexer-score-width-pr45794-2026` | `sgl-project/sglang` `65f19fce48` (PR #45794) | Aligns DeepSeek-V4.1 BF16 prefill indexer score width and derives rank-local CP `req` indices from `rows_per_request_device` when `inputs.req_rows` is omitted. |
| `sglang-diffusion-wan-normed-qk-direct-ulysses-exchange-pr46247-2026` | `sgl-project/sglang` `cf66f7ed73` (PR #46247) | Writes Wan diffusion normed+RoPE'd Q/K straight into `[M, block_cols]` Ulysses IPC all-to-all exchange slots via `fused_rmsnorm_interleaved_rope_into_blocks` and `fill_all`. |
| `vllm-preserve-fp16-legacy-fused-moe-lora-pr59931-2026` | `vllm-project/vllm` `2f1cb151fd` (PR #59931) | Casts both `tl.dot` operands to matching `dtype` in `_fused_moe_lora_kernel` so legacy `tl.load` paths preserve `float16` instead of forcing `bfloat16`. |
| `vllm-multimodal-release-consumed-mrv2-cpu-inputs-pr57368-2026` | `vllm-project/vllm` `373614f66d` (PR #57368) | Releases consumed Model Runner V2 (`MRV2`) CPU multimodal feature tensors immediately after encoder execution. |
| `vllm-kv-connector-decode-bench-report-all-groups-loaded-pr62242-2026` | `vllm-project/vllm` `beeed65c10` (PR #62242) | Implements `DecodeBenchConnector.get_loaded_kv_cache_group_ids()` returning all `transfer_group_ids` for hybrid attention + sliding-window models. |
| `trtllm-unify-multimodal-request-data-item-metadata-pr20391-2026` | `NVIDIA/TensorRT-LLM` `9ffa6f4395` (PR #20391) | Unifies multimodal preprocessing and PyTorch executor runtime around `MultimodalParams` and per-item metadata. |

## 2. Mathematical Modeling & Architect Exercises

1. **Wan Direct-into-Blocks Ulysses Exchange Traffic Reduction**:
   For $M = 4096, H = 32, D = 128, U = 4$ (`block_cols = 1024`), writing normed+RoPE'd Q/K straight into the `[M, block_cols]` Ulysses IPC slots eliminates the intermediate full `[M, H*D]` write+read roundtrip (`256.0 MiB -> 128.0 MiB`, a **50.0% HBM traffic reduction**).
2. **vLLM MRV2 Host Memory Reclamation**:
   Releasing consumed MRV2 CPU inputs across 64 concurrent multimodal requests (`8.0 MiB` each) drops retained post-encoder host memory from `512.0 MiB` to `0.0 MiB`.

## 3. Reproducible Offline Verification

```bash
python experiments/sglang_beam_dsv4_bridges_kimi_tgv_dsv41_table_wan_ulysses_vllm_lora_mrv2_trtllm_phase96/run.py --check
```
Verified in `summary.json`: `stale_graph_pointer_reads == 0`, `hbm_traffic_saving_fraction == 0.5`, `released_after_encoder_cpu_mb == 0.0`, and `all_groups_loaded == True`.
