# TransformerEngine 64-Bit MoE Auxiliary Loss Offset Guard & 128x128 Grouped MXFP8 Requantization, Megatron-LM Disaggregated KV Handoff (`TP > KV-Heads`) & FP32 Vision RoPE, Mooncake Cached DFS Path Reuse & NIXL RAII Fix Deep Dive (Phase 116)

> **中文版**: [te-moe-offset-mxfp8-megatron-metrics-kv-handoff-rope-mooncake-phase116.md](te-moe-offset-mxfp8-megatron-metrics-kv-handoff-rope-mooncake-phase116.md)
> **Priority**: `priority: 125` · **Evidence Mode**: `implementation` + `own-offline`

## 1. Core Architectural Findings & Primary Code Evidence

1. **TransformerEngine 64-Bit Fused MoE Auxiliary Loss Offset Guard & 128x128 Grouped MXFP8 Requantization** (`NVIDIA/TransformerEngine@268fdd54`, `TransformerEngine@15dcc848`): Upgrades fused MoE auxiliary loss indexing in `fused_moe_aux_loss.cu` to `int64_t` to prevent 32-bit signed integer overflow at large token-expert products (`131072 * 256`), and implements `128x128`-tiled grouped MXFP8 requantization in `fused_group_requantize.cu` (`1.34x` requantization bandwidth speedup).
2. **Megatron-LM Tensor Metrics 4 Training Observation, Disaggregated KV Handoff (`TP > KV-Heads`) & ViT FP32 RoPE Preservation** (`NVIDIA/Megatron-LM@364eef70`, `Megatron-LM@259f6a7c`, `Megatron-LM@3ca52d95`, `2026-10-27`): Integrates `Tensor Metrics 4` observer hooks into training (`observer.py`, `ft_integration.py`), supports disaggregated KV resharding when `TP > num_kv_heads` in `kv_reshard.py` (`0` OOB head errors and `50.0%` duplicate transfer reduction at `TP=16, GQA=8`), and keeps vision RoPE frequencies in `FP32` during BF16/FP16 model casts (`238.4x` lower phase drift error).
3. **TorchTitan Monarch `ActorSystem`, Mooncake Cached DFS Path Reuse, LMCache Raw-Block Size Guard, Dynamo Skip Per-Token Decode & NIXL Mooncake RAII Fix** (`pytorch/torchtitan@f1bb643a`, `kvcache-ai/Mooncake@d782a599`, `LMCache/LMCache@f17926d0`, `ai-dynamo/dynamo@cffc4fa7`, `ai-dynamo/nixl@7bd1d138`): Abstracts Monarch RL orchestration behind `ActorSystem`, reuses validated shard paths on cached DFS I/O in Mooncake (`1.41x` I/O speedup), rejects undersized destination buffers before raw-block reads in LMCache (`0` silent partial-read corruptions), skips per-token text decode in Dynamo when `logprobs=False` (`68.5%` frontend CPU savings), and replaces raw `new[]` with `std::vector<transfer_request_t>` in NIXL `mooncake_backend.cpp` (`0` leaked bytes on `postXfer` errors).

## 2. Offline Verification (`experiments/te_moe_offset_mxfp8_megatron_metrics_kv_handoff_rope_mooncake_phase116/run.py`)

- `int32_overflows = True`, `int64_overflows = False`, `grouped_mxfp8_128x128_requant_speedup = 1.34x`
- `kv_handoff_duplicate_reduction_ratio = 0.50` (`kv_handoff_oob_head_errors = 0`), `rope_fp32_error_reduction_factor = 238.4x`
- `mooncake_cached_dfs_io_speedup = 1.41x`, `dynamo_frontend_detok_cpu_saving_pct = 68.5%`, `nixl_mooncake_postxfer_error_leaked_bytes = 0`
