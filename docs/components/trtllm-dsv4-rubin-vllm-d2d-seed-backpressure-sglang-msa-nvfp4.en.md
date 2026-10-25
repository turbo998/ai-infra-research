# TensorRT-LLM DeepSeek-V4 Rubin (SM107) In-Place MXFP8 o_b_proj & CuteDSL MoE Async Input Quant, vLLM D2D Warmup Cache Seeding & Offloading EMA Failure Backpressure, SGLang MiniMax M3.1 Sparse NVFP4 Attention (MSA-v1) & 6-Vendor Pareto Frontier

> **中文版本**: [trtllm-dsv4-rubin-vllm-d2d-seed-backpressure-sglang-msa-nvfp4.md](trtllm-dsv4-rubin-vllm-d2d-seed-backpressure-sglang-msa-nvfp4.md)  
> **Question ID**: `trtllm-dsv4-rubin-vllm-d2d-seed-backpressure-sglang-msa-nvfp4` (Phase 74, Priority 83)  
> **Reproducible Experiment**: [`experiments/trtllm_dsv4_rubin_vllm_d2d_seed_backpressure_sglang_msa_nvfp4_sim/run.py`](../../experiments/trtllm_dsv4_rubin_vllm_d2d_seed_backpressure_sglang_msa_nvfp4_sim/run.py)  
> **Evidence Class**: `implementation` / `author-benchmark` (upstream commits & PR benchmarks) + `own-offline` (deterministic stdlib simulator)

---

## 1. Architectural Motivation & Bottlenecks

In production serving of sparse-attention MoE models such as DeepSeek-V4 / DSpark and MiniMax M3.1, six engine-level bottlenecks were addressed across TensorRT-LLM, vLLM, and SGLang between `2026-10-24` and `2026-10-25`:

1. **DeepSeek-V4 on Rubin (`SM107`) Output Copy Overhead & Redundant `q_pe` RoPE** ([`TensorRT-LLM@d7e0ecac925d`](https://github.com/NVIDIA/TensorRT-LLM/commit/d7e0ecac925dc8f3fddfc2bff5ef5ab1d9149fbf)): `o_b_proj` MXFP8 linear layers allocated a temporary output tensor before copying into `_rubin_dsv4_o_b_proj_out`, lazily allocated `_quant_scale_qkv` during first CUDA-graph capture (causing an H2D copy), and re-ran `q_pe` RoPE even when `q_b` GEMM had already emitted rotated FP8 Q.
2. **CuteDSL MoE Async Input Quantization Host Overhead in Single-Stream Eager Mode** ([`TensorRT-LLM@032eb456ecb1`](https://github.com/NVIDIA/TensorRT-LLM/commit/032eb456ecb145ba10f9c9dc043cfaa8254d657c)): Forking `quantize_input_async` onto an auxiliary stream overlaps quantization with `top-k` routing and `moe_sort` under CUDA graphs (saving `~25 us/layer` or `1.525 ms/step` across 61 MoE layers at `batch=8` on GB300), but incurs host stream-switch/event overhead when run eagerly without multi-stream enabled.
3. **vLLM Cold-Start Autotuning Sweeps on Scale-Out Preload Pods** ([`vllm@94a29d316e8c`](https://github.com/vllm-project/vllm/commit/94a29d316e8c1d6dcf6ff75645c2e61c7e3c83fd)): New `--warmup-level preload` replicas previously re-ran full DeepGEMM, FlashInfer, Triton, and TorchCompile autotuning sweeps (`~148 s`) instead of inheriting warm artifacts from running peers.
4. **vLLM Offloading Storage-Full (`ENOSPC`) Blind Spot & Preemption Re-Store Skip** ([`vllm@2b1f75c80f79`](https://github.com/vllm-project/vllm/commit/2b1f75c80f79496b313e28362f0856632dfbf5ea), [`vllm@6f51433838b8`](https://github.com/vllm-project/vllm/commit/6f51433838b858a9cd0478b2ffee4d2c610a1712)): Fast-failing `fs`/`obj` writes (`ENOSPC`) reported near-zero duration and bypassed latency-only `EMABackpressureDetector`; furthermore, preempted requests restarting at `0` computed tokens kept a stale `next_stored_chunk_idx > 0` and never re-stored evicted prefix chunks.
5. **SGLang MiniMax M3.1 Sparse NVFP4 Attention (`MSA-v1`) & FP8 Double-Rounding** ([`sglang@4991bd33ae1c`](https://github.com/sgl-project/sglang/commit/4991bd33ae1c9d7468394eaf966d7db88f352b7b)): MiniMax M3.1 combines Q8KV4 attention with a K-only FP8 sparse index (`64` Q heads, `4` KV heads, `4` index heads, `block_size=128`, `topk_blocks=16`), requiring strict BF16-rounded FP8 casting in `fused_gemma_qknorm_rope_fp8`.
6. **SGLang DeepSeek-V4 `FLASHINFER_MEGAMOE` Shared-Expert Addition Bug** ([`sglang@ad950cdcd966`](https://github.com/sgl-project/sglang/commit/ad950cdcd9666a104321deeaf1cbf883b4e8c60b)): `MoeTransport.prepare_output` checked only `self.rows.group is not None` to set `summed=True`, causing `FLASHINFER_MEGAMOE` (`self.rows.group is None`, `is_complete=True`) to skip shared-expert addition on non-zero TP ranks.

---

## 2. Core Implementation Mechanics

### 2.1 TensorRT-LLM DeepSeek-V4 Rubin (`SM107`) In-Place `o_b_proj` & MoE Async Quant Gate

- **In-Place `o_b_proj` & Pre-Registered QKV Scale** (`d7e0ecac925d`): Adds `allow_unpartitioned_output=True` to `CuteDSLBlockScaledRubinLinear` so `_rubin_dsv4_o_b_proj_out` is written in-place, pre-registers `_quant_scale_qkv` at initialization, and passes `q_rope_applied = fuse_q_fp8_in_rope && !q_pe.has_value()` to skip redundant `q_pe` RoPE writes.
- **MoE Multi-Stream Async Input Quant Gate** (`032eb456ecb1`): Gates `_should_quantize_input_async` on `moe.comm is None and not moe.apply_router_weight_on_input and (do_multi_stream() or is_current_stream_capturing())`, saving **`1.525 ms/step`** across 61 MoE layers at `batch=8` on GB300 under CUDA graphs while avoiding `0.58 ms/step` host stream-fork overhead in eager mode.

### 2.2 vLLM D2D Warmup Cache Seeding & Failure-Aware EMA Backpressure

- **D2D Warmup Cache Seeding** (`94a29d316e8c`): Streams compiled artifacts (`torch_compile`, `triton`, `deep_gemm`, `flashinfer`) over `base_port + group_offset + rank` with `--seed-token` authentication and GPU/CUDA/cuDNN metadata validation, cutting scale-out warmup from `148.0 s` to **`3.2 s` (`46.25x`)**.
- **Failure-Aware EMA Backpressure & Preemption Rewind** (`2b1f75c80f79`, `6f51433838b8`): After `DEFAULT_FAILURE_THRESHOLD = 3` consecutive store failures on `fs`/`obj` media (`p2p` exempted), `EMABackpressureDetector` sets `ema = 4.0 * high_water_s = 0.200 s`, driving `ThrottledDropPolicy` (`ramp_factor=2.0`) to **`100%` drop probability** with idle exponential recovery; preempted requests reset `group_state.next_stored_chunk_idx = 0`.

### 2.3 SGLang MiniMax M3.1 `MSA-v1` Sparse NVFP4 Attention & DSv4 MegaMoE Fix

- **`MiniMaxSparseNvfp4AttnBackend` (`MSA-v1`)** (`4991bd33ae1c`): Packs NVFP4 K+V (`144 B/head/token`) + K-only FP8 index (`132 B/head/token`) into `1,104 B/token` across 4 KV heads (**`1.855x`** smaller than `2,048 B` BF16) and reads only `16 × 128 = 2,048` selected KV tokens + `1,024` index tokens at `131,072` context (**`156.04x`** lower read bandwidth than dense BF16).
- **Bit-Exact FP8 Cast & MegaMoE Shared-Expert Guard** (`4991bd33ae1c`, `ad950cdcd966`): Casts `fp8_e4m3_t` from rounded `o_bf16` and updates `MoeTransport.prepare_output` to recognize `self.rows.group is None and post_experts_output_is_complete(is_tp_path=False)`.

---

## 3. Quantitative Summary & 6-Vendor Pareto Frontier

Run `python experiments/trtllm_dsv4_rubin_vllm_d2d_seed_backpressure_sglang_msa_nvfp4_sim/run.py --check`:

| Platform / Architecture | $/1M Tokens (USD) | P99 TTFT (ms) | Decode TPS / User | Key Phase 74 Architectural Lever |
|---|---|---|---|---|
| **NVIDIA Vera Rubin VR200 (NVL72, SM107)** | **$0.09** | `175.0` | `148.0` | In-place MXFP8 `o_b_proj` + pre-registered QKV scale + skipped `q_pe` RoPE |
| **NVIDIA Blackwell Ultra GB300 (NVL72, SM103)** | **$0.14** | `215.0` | `118.0` | CuteDSL MoE multi-stream async input quant (`-1.525 ms/step`) + QSA walk |
| **Fireworks AI (Blackwell Disagg + D2D Seed)** | **$0.18** | `240.0` | `106.0` | D2D warmup cache seeding (`46.25x` faster scale-out) + failure EMA shedding |
| **Together AI (Turbo Spec + MSA-v1 NVFP4)** | **$0.19** | `250.0` | `102.0` | MiniMax M3.1 `MSA-v1` Q8KV4 sparse attention (`156.04x` 128K read reduction) |
| **Groq LPU V2 (Deterministic SRAM Pipeline)** | **$0.58** | `85.0` | `310.0` | On-chip SRAM zero-HBM jitter for short-context ultra-low latency |
| **Cerebras WSE-3 (Wafer-Scale Swarm)** | **$0.62** | `92.0` | `420.0` | Wafer-scale single-user decode speed, bounded by on-wafer capacity/cost |
