# FlashInfer Frost DeepSeek-V4.1 SM100 Mixed MXFP8/FP4 Sparse Decode (`BlackwellV41MixedCacheDecode`), Non-Interleaved Page Scale Layout, and CUDA Green Context SM Partitioning (`green_ctx.py`)

> **中文版**: [flashinfer-frost-ds41-mixed-mxfp8-fp4-decode.md](flashinfer-frost-ds41-mixed-mxfp8-fp4-decode.md) | **Experiment**: [`experiments/flashinfer_ds41_mixed_decode_sim/README.md`](../../experiments/flashinfer_ds41_mixed_decode_sim/README.md)

### Executive Summary

FlashInfer's **Frost DeepSeek-V4.1 mixed-cache decode kernel (`deepseek_v41_decode` / `BlackwellV41MixedCacheDecode`)** on SM100 (NVIDIA Blackwell) fuses a 128-token local sliding-window KV cache (`swa_mxfp8`, `[pages, 64, 1, 528]`, E4M3 data + group32/E8M0 power-of-two scales) with a $K \in \{64, \dots, 512\}$ compressed sparse global KV cache (`main_kv_fp4`, `[pages, 64, 1, 288]`, E2M1x2 packed data + group16/E4M3 scales) inside a single CuTe DSL kernel. By combining a **non-interleaved 64-token page layout** (all data rows followed by all scale rows per page), **round-to-nearest division (`tl.div_rn`)** prior to E2M1 conversion, and an adaptive schedule transitioning from small-batch Split-KV to large-batch 16-warp Warp-Specialization (`use_ws`) with wave-balanced persistent CTAs (`persistent_ctas = ceil(B / ceil(B / sm_count))`), it shrinks 640-token active KV bytes read per layer from `655,360 B` (BF16) to `215,040 B` (**3.0476x compression / 67.19% HBM bandwidth reduction**), while `flashinfer/green_ctx.py` provides `(8, 8)` SM-aligned CUDA Green Context streams for interference-free kernel co-scheduling.

### Upstream Source Responsibility & Locations

| Module & File | Upstream Repo & Commit / Blob SHA | Core Engineering Responsibility |
| :--- | :--- | :--- |
| `flashinfer/deepseek_v41.py` | `flashinfer-ai/flashinfer` commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5` / blob `52ec7df5577024e57993cb95e4832dc653a7b3e7` | Exposes `deepseek_v41_decode`, `deepseek_v41_window_decode`, and `deepseek_v41_quantize_cache`, defining Q/SWA/Global/Sink contracts, CUDA Graph fixed-address replay rules, and sink-exclusive natural-log LSE semantics. |
| `flashinfer/experimental/deepseek_v41/decode.py` | `flashinfer-ai/flashinfer` commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5` / blob `e0bdf9529778bbef5eb8c6aa82811bd136c0051e` | Implements `_Plan` ownership, 16-byte alignment and non-overlap checks, small-batch Split-KV (`desired = {10, 8, 4, 2, 1}`), and large-batch 16-warp WS + wave-balanced `persistent_ctas` compilation. |
| `flashinfer/experimental/deepseek_v41/cache.py` | `flashinfer-ai/flashinfer` commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5` / blob `3181b6636b22f6efef02808703b8d04392075ca0` | Implements the `_quantize_cache` Triton kernel producing non-interleaved 64-token pages (`swa_mxfp8` 528 B/tok and `main_kv_fp4` 288 B/tok) using `tl.div_rn` and inline PTX `cvt.rn.satfinite.e2m1x2.f32`. |
| `flashinfer/green_ctx.py` | `flashinfer-ai/flashinfer` commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5` / blob `66b5970dae4d8e6881217b675368d072a0033494` | Wraps CUDA Driver APIs `cuDevSmResourceSplitByCount`, `cuGreenCtxCreate`, and `cuGreenCtxStreamCreate` with `(8, 8)` SM granularity on SM90/SM100. |

### Core Implementation Path, Memory Layout & Synchronization Contracts

1. **Non-Interleaved 64-Token Page Layout (`cache.py:25-70`)**:
   - Each 64-token page stores all 64 quantized data rows contiguously first, followed by all 64 scale rows contiguously:
     - `swa_mxfp8` (`uint8 [pages, 64, 1, 528]`): `WIDTH = 512` (512 E4M3 bytes), `GROUP = 32`, `SF = 16` (16 E8M0 power-of-two exponent bytes). Data occupies bytes `[0, 32768)` and scales occupy `[32768, 33792)` within each `33,792 B` page.
     - `main_kv_fp4` (`uint8 [pages, 64, 1, 288]`): `WIDTH = 256` (512 E2M1 elements packed 2-per-byte), `GROUP = 16`, `SF = 32` (32 E4M3 group scales). Data occupies bytes `[0, 16384)` and scales occupy `[16384, 18432)` within each `18,432 B` page.
2. **Why `tl.div_rn` Is Mandatory Before E2M1 Conversion (`cache.py:42-53`)**:
   - Reciprocal multiplication (`x * (1.0 / scale)`) perturbs exact E2M1 halfway values (`{0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}`) when `scale` is a non-power-of-two E4M3 float (such as `0.9375`), shifting `1.25` to `1.25000012` and causing `cvt.rn.satfinite.e2m1x2.f32` to jump from `1.0` to `1.5`. `cache.py` enforces round-to-nearest division via `tl.div_rn(ga, scale[:, :, None])` with `enable_fp_fusion=False`.
3. **Adaptive Split-KV, Warp-Specialization & Wave-Balanced Persistent CTAs (`decode.py:130-195`)**:
   - Small batches (`B <= 64`) use Split-KV (`split = min(desired, (128 + ck) // 64)`) with workspace `B * 64 * split * 513 * 4` bytes.
   - Large batches (`ck == 512, B >= 128`) switch to `split = 1`, `dequant_warps = 16`, `phase_chunks = 2`, `use_ws = True`, and balance multi-wave grids via `waves = ceil(B / sm_count)` and `persistent_ctas = ceil(B / waves)`.
4. **Sink-Exclusive Natural-Log LSE & Green Context Alignment (`hca_v41.py:66-76`, `green_ctx.py:30-115`)**:
   - Public `lse` (`FP32 [B, 64, 1]`) excludes the sink logit (`FP32 [64]`); the sink is applied once at final normalization.
   - `green_ctx.py` enforces `(min_count=8, granularity=8)` on SM90/SM100 (`major >= 9`).

### Key Equations, Units & Engineering Assumptions

$$\text{base}(\text{slot}) = \left\lfloor \frac{\text{slot}}{64} \right\rfloor \times 64 \times (\text{WIDTH} + \text{SF}), \quad \text{data\_base} = \text{base} + (\text{slot} \bmod 64)\cdot\text{WIDTH}, \quad \text{scale\_base} = \text{base} + 64\cdot\text{WIDTH} + (\text{slot} \bmod 64)\cdot\text{SF}$$

$$W = \left\lceil \frac{B}{\text{SM}_{\text{count}}} \right\rceil, \quad \text{CTA}_{\text{persistent}} = \left\lceil \frac{B}{W} \right\rceil, \quad \text{LSE}_{\text{real}} = \ln \left( \sum_{j \in \mathcal{S}_{\text{swa}} \cup \mathcal{S}_{\text{global}}} e^{s_j} \right)$$

### Deterministic Offline Simulation & Quantitative Comparison

Validated via [`experiments/flashinfer_ds41_mixed_decode_sim/run.py`](../../experiments/flashinfer_ds41_mixed_decode_sim/run.py) and [`experiments/flashinfer_ds41_mixed_decode_sim/results.json`](../../experiments/flashinfer_ds41_mixed_decode_sim/results.json):

| Dimension | Metric / Scenario | Baseline (BF16 / Reciprocal / Full Grid) | Frost DS4.1 (`swa_mxfp8` + `main_kv_fp4`) | Quantitative Finding |
| :--- | :--- | :--- | :--- | :--- |
| **KV Bytes per 640 Active Tokens** | 128 SWA + 512 Global | `655,360 B` (BF16 `1024 B/tok`) | `215,040 B` (`128*528 + 512*288`) | **3.0476x compression** (**67.19%** bandwidth saved) |
| **E2M1 Halfway Boundary Accuracy** | 217 E4M3 scale tests | `x * (1.0 / scale)` perturbs **9 cases (4.15%)** | `tl.div_rn(x, scale)` **0 perturbations** | Eliminates up to **+50%** step error at E2M1 ties |
| **Tail-Wave Balancing (`B=160, SM=148`)** | 2-wave grid utilization | Naive grid runs 12 CTAs in wave 2 (**54.05%**) | `persistent_ctas = 80` (**100.0%** balanced) | Full wave balance with `0 B` Split-K workspace |
| **CUDA Green Context Split (`SM=148`)** | Request `[20, 36, 64]` SMs | Unaligned SM partitioning | Aligned to `[24, 40, 64]` SMs (`20` remaining) | Enforces `(8, 8)` hardware cluster alignment |

### Architectural Impact on Quality, SLO, Topology, Cost & Operations

- **Quality & Bandwidth**: Keeps recent 128 sliding-window tokens in MXFP8 while compressing 512 sparse global tokens into FP4, cutting KV HBM traffic by 67.19% without degrading local context fidelity.
- **CUDA Graph Contract**: Prepare `_Plan` before graph capture; keep input pointers fixed across replays or update values in place; never run the same `_Plan` concurrently across multiple streams.
