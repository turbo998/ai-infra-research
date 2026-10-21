# FlashInfer Cake NVFP4-Transport All-Reduce + RMSNorm Per-Architecture Programs, MNNVL One-Shot FP32 Square Fix, Sage-FP8 Block-Sparse Attention, TRT-LLM-Gen Caller-Owned MoE Workspace & Kimi-K3 SiTU Folding

> **Updated**: 2026-10-22 · **Phase**: Phase 53 (`priority: 62`) · **Category**: `operator-kernel`  
> **Deterministic Experiment**: [`experiments/flashinfer-cake-fp4-ar-mnnvl-fp32-sage-fp8-bsa-trtllm-moe-ws-k3-situ/run.py`](../../experiments/flashinfer-cake-fp4-ar-mnnvl-fp32-sage-fp8-bsa-trtllm-moe-ws-k3-situ/run.py)

---

## 1. First-Hand Upstream Sources

| Repository & Commit | Date (UTC) | Core Engineering Contribution |
|---|---|---|
| [`flashinfer-ai/flashinfer@067b36956d`](https://github.com/flashinfer-ai/flashinfer/commit/067b36956da21f9cb6cfbdad9fd6326c52fa3655) | 2026-10-21 | `perf(cake_comm)`: Per-architecture (`sm_100a`/`sm_103a`/`sm_107a`) programs (`_rso`, `_rst`, `_qp`, `_sop`, `_pf`) and permuted pass-1 sum of squares for NVFP4-transport all-reduce + RMSNorm (#6954) |
| [`flashinfer-ai/flashinfer@1ff146212d`](https://github.com/flashinfer-ai/flashinfer/commit/1ff146212d7f0e5641c361f316a7f039633f98df) | 2026-10-21 | `fix(comm)`: Cast to `float` before squaring in MNNVL one-shot fused all-reduce + RMSNorm to prevent FP16 overflow (`|x| > 255.94 -> inf -> 0`) (#6818) |
| [`flashinfer-ai/flashinfer@4e4c41da67`](https://github.com/flashinfer-ai/flashinfer/commit/4e4c41da67f6c463940e4c75d11207ce93d6c516) | 2026-10-20 | `perf(cake_sage)`: Single-launch SM100/SM103 Sage-FP8 block-sparse attention (`1.30x–1.40x`) and fused quantizers (`1.23x`) (#6872) |
| [`flashinfer-ai/flashinfer@15c83f1c36`](https://github.com/flashinfer-ai/flashinfer/commit/15c83f1c36f5468c6fdc49e8f7f8ede247705cbb) | 2026-10-20 | `feat(moe)`: `trtllm_fused_moe_workspace_size` and caller-owned `workspace_buffer=` cutting B300 autotuning peak memory by `2,683.1 MiB` (#6825) |
| [`flashinfer-ai/flashinfer@75b30ffdbf`](https://github.com/flashinfer-ai/flashinfer/commit/75b30ffdbf8c4413b26ff12ba7cb4c6c7d5f4d46) | 2026-10-20 | `deslop(cake_megamoe)`: Drop duplicate 1,090-line SM120 `part002.cuh`, re-seal 18 manifest digests, and remove uncalled EP16 host helper (#6898) |
| [`flashinfer-ai/flashinfer@e2456a4eea`](https://github.com/flashinfer-ai/flashinfer/commit/e2456a4eeafe3ab14159b8debebf0582e34d3ed4) | 2026-10-20 | `deslop(cake_kimi_k3_situ)`: Extract 103 guarded device helpers into one header, fold `fc1:n8_s2a`, and share 8 host bindings (`-17,018` lines) (#6879) |

---

## 2. Deep-Dive Architectural Analysis

### 2.1 Per-Architecture NVFP4-Transport All-Reduce + RMSNorm (`067b36956d`)

`flashinfer.comm.cake_fp4_allreduce` expands its program registry to **37 programs (20 per architecture across `sm_100a`, `sm_103a`, and `sm_107a`; 22 new, 16 retired)** using architecture-specific kernel schedules:
- **`_rso` (`sm_103a` TP2 one-shot BF16)**: Reads the peer packet's per-row scale word in flight during the first data sweep (`12.99 -> 12.04 us`, **`1.0789x`** at `1K x 6144` on GB300).
- **`_rst` (`sm_107a` TP2 one-shot BF16)**: Resets the consumed Lamport slot with per-thread stores instead of a bulk store (**`1.0121x`** geomean).
- **`_qp` (`sm_100a`/`sm_103a` fused BF16 -> NVFP4/MXFP8)**: Replaces the scalar FP32 quantization tree in pass 2 with packed two-wide arithmetic.
- **`_sop` + `_pf` (permuted pass-1 sum of squares + 4K residual prefetch)**: Accumulates squares in each lane's loaded chunk order, achieving a **`1.0718x`** geomean speedup on B200 (`sm_100a`) and **`1.0300x–1.0718x`** on GB300 (`sm_103a`), peaking at **`1.0942x`** (`77.36 -> 70.70 us` at `8K x 6144` `NVFP4 -> NVFP4`).

### 2.2 MNNVL One-Shot Fused All-Reduce + RMSNorm FP32 Square Fix (`1ff146212d`)

In `include/flashinfer/comm/trtllm_mnnvl_allreduce.cuh` (`oneshot_allreduce_fusion_kernel`), computing `toFloat<T>(packedAccum.elements[i] * packedAccum.elements[i])` evaluated `x * x` in `T`. For `half` (FP16), any activation with $|x| > \sqrt{65504} \approx 255.94$ overflowed to `inf`, turning `rsqrtf(inf)` into `0` and zeroing out the entire token. In Qwen3.5-27B FP16 TP=2 under vLLM 0.25.1, ~`9.5%` of decode tokens exceeded `256` at the final norm, collapsing generation into `!!!!` (`0.000` RL validation score). Casting to `float` before squaring restores the RL validation score to **`0.698`**.

### 2.3 SM100/SM103 Sage-FP8 Block-Sparse Attention & Quantizers (`4e4c41da67`)

`flashinfer/cake_sage_block_sparse_attention_backend` merges descriptor upload and kernel execution into a single launch on SM100/SM103 (`head_dim=128`, E4M3 Q per-token / K per-16-token / V per-channel scales, 64x64 blocks), achieving **`1.30x–1.40x`** faster attention forward (`0.0219 -> 0.0156 ms` on `A4`, `1.8138 -> 1.3691 ms` on `A3`) and **`1.23x`** faster fused `quantize_qk_vamax` + `quantize_v` (`0.2610 -> 0.2127 ms`) on GB300 (`sm_103a`).

### 2.4 TRT-LLM-Gen Caller-Owned MoE Workspace & Kimi-K3 SiTU Folding (`15c83f1c36`, `75b30ffdbf`, `e2456a4eea`)

- **`trtllm_fused_moe_workspace_size` (`15c83f1c36`)**: Adds a `query_workspace_size` dry-run mode and `workspace_buffer=` reuse across 11 quantization modes, cutting B300 autotuning peak memory on `16384 tokens, H=7168, I=2048, E=256, top-8` from **`3,579.9 MiB` to `896.8 MiB` (`-2,683.1 MiB` / `-74.95%`)**.
- **SM120 MegaMoE & Kimi-K3 SiTU Deslop (`75b30ffdbf`, `e2456a4eea`)**: Removes the byte-identical 1,090-line SM120 `part002.cuh` with 18 verified manifest seals, extracts 103 guarded (`CAKE_SITU_NEED_*`) device helpers into `cake_kimi_k3_nvfp4_situ_device.cuh`, folds the `fc1:n8_s2a` SM100/SM103 pair, and shares 8 host bindings via `CAKE_SITU_KERNEL`/`CAKE_SITU_SHIM`, shrinking `csrc/fused_moe/cake_kimi_k3_situ/` from **`75,311` to `58,293` lines (`-17,018` lines / `-22.60%`)** with identical SASS across all 73 builds.
