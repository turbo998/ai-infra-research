# Source Audit of FlashInfer `cake_warp_decode` K256 FC2, Rubin (`sm_107a`) MLA Dual-Tile Epilogue & `cake_mok` / `cake_selective_state_update`

> **中文版**: [flashinfer-warp-decode-k256-fc2-rubin-mla-pairpipe-mok-load-bins-ssm-update.md](flashinfer-warp-decode-k256-fc2-rubin-mla-pairpipe-mok-load-bins-ssm-update.md) | **Question ID**: `q-flashinfer-warp-decode-k256-fc2-rubin-mla-pairpipe-mok-load-bins-ssm-update` (`priority: 91`, Phase 82) | **Reproducible Experiment**: `experiments/flashinfer_warp_decode_k256_fc2_rubin_mla_pairpipe_mok_load_bins_ssm_update/run.py`

---

## 1. Architectural Context & Core Engineering Bottlenecks

Across Blackwell (`sm_100a` B200/GB200), Blackwell Ultra (`sm_103a` GB300), and Rubin (`sm_107a` R200) GPUs, routed MoE decode, Multi-head Latent Attention (MLA) NVFP4 paged decode, and Mamba2 Selective State Update (SSM) kernels confront three microarchitectural bottlenecks:

1. **FC2 Tail Latency in High-Expert (`E=512, top-10`) Narrow-Intermediate (`I=256/512`) MoE Decode**: In `cake_warp_decode` (`1..32` tokens, NVFP4 `W4A4` with BF16 output), when `H=4096, E=512, K=10` and `T >= 7..19`, throttled work queues and narrower K-tiles incur excessive MMA loop iterations and scheduling barriers per 128-row output tile.
2. **Rubin (`sm_107a`) MLA NVFP4 Paged Decode PV Epilogue TMA Store Stalls & Serial Converter Rounds**: In `cake_rubin_mla_nvfp4_decode` (`352 B/token` paged NVFP4 KV cache), using a single `4 KB` epilogue staging tile per correction warp forced each of the eight 64-dim PV output rounds to wait for the immediately preceding round's bulk tensor store; meanwhile, full-decode converter warps serially waited for `16B` loads, scale construction, and conversion in each round.
3. **Host Launch Overhead in `cake_mok` & SM107 `tcgen05` Descriptor Address Mask Divergence in Mamba2 SSD**: `cake_mok` recomputed expert layouts on every forward/recompute launch and held dead backward scratch buffers across the output epilogue, while SM107 widened the `tcgen05` shared-memory descriptor start-address field from **14 bits (`0x3FFF` on SM100/SM103)** to **15 bits (`0x7FFF` on SM107)**, previously duplicating 21 device `.cu` renders.

---

## 2. First-Hand Source Code Implementation Audit

### 2.1 `cake_warp_decode` 1-D Linear-Grid Unthrottled `K256 FC2` (`53f6bc4d3f47`, `#7339`)

Auditing `csrc/fused_moe/warp_decode/cake_warp_decode_contract.cuh`, `generated/cake_warp_decode_generated_manifest.cuh`, and `generated/common/cake_warp_decode_4f408b642a6c28d2940e_kernel.cu`:

- **Launch Contract Extension (`CakeWarpDecodeLaunchContract`)**: Adds `kLaunchLinearWorkfeedUnthrottled = 3u`, mapped in `cake_warp_decode_launch_grid()` to a 1-D linear CTA grid `dim3(cta_count, 1, 1)` with an unthrottled atomic work queue.
- **`K256 FC2` Kernel (`kernel_cake_warp_decode_4f408b642a6c28d2940e`)**: Configured with `__launch_bounds__(256, 1)` and **`179,456 B` (`175.25 KiB`) dynamic shared memory**. It executes one fused `K=256` NVFP4 block-scaled MMA per 128-row output tile of an 8-token route tile, producing bit-identical outputs to the previous kernel.
- **Exact Manifest Routing**:
  - On `sm_100a` (GB200): routes `h4096_e512_i256_k10` for `num_tokens = 9..32` (`24` rows) and `h4096_e512_i512_k10` for `num_tokens = 19..32` (`14` rows), totaling **`38` rows**.
  - On `sm_103a` (GB300): routes `h4096_e512_i256_k10` for `num_tokens = 7..32` (`26` rows) and `h4096_e512_i512_k10` for `num_tokens = 12..32` (`21` rows), totaling **`47` rows**.
- **Author-Reported Hardware Performance (`author-benchmark`, GB200 `sm_100a` & GB300 `sm_103a`, CUDA 13.4 cold L2)**:
  - On `sm_100a`, `h4096_e512_i256_k10` at `T=32` (read-flush cold L2) drops from `98.18 us` to **`93.46 us`** (ratio `0.952`, **`1.0505x` speedup**), and from `103.84 us` to **`99.38 us`** under memset-flush cold L2.
  - On `sm_103a`, `h4096_e512_i256_k10` at `T=31` (read-flush cold L2) drops from `95.57 us` to **`91.24 us`** (ratio `0.955`, **`1.0475x` speedup**).
  - `h4096_e512_i512_k10` at `T=32` on `sm_100a` improves from `151.61 us` to **`148.80 us`** (**`1.0189x` speedup**).

### 2.2 Rubin (`sm_107a`) MLA NVFP4 Paged Decode Dual Staging Tiles & `decode_full_pairpipe` (`5c314c510256`, `#7329`)

Auditing `flashinfer/experimental/cake_rubin_mla_nvfp4_decode/{build.json,cake_backend.py,README.md}`:

- **Two 4 KB Epilogue Staging Tiles per Correction Warp**: Across the eight 64-dim PV output rounds, each correction warp alternates between **two `4 KB` shared-memory staging tiles**, waiting only for the TMA bulk tensor store from two rounds back (`r - 2`). Dynamic shared memory increases from **`323,584 B` (`316 KiB`) to `332,800 B` (`325 KiB`, `+9,216 B`)**.
- **`decode_full_pairpipe` Converter Software Pipeline**: Adds `decode_full_pairpipe` (`csrc/decode_full_pairpipe.cu`), where the converter warp preloads the `16-byte` pieces and scale words for all eight rounds of a tile before forming scales and converting. `cake_backend.py` dispatches `decode_full_pairpipe_run` for `H96-Q8-B64-KV64K` (`num_heads == 96, q_len == 8, batch_size == 64, kv_len in (65536, 98304)`).
- **Author-Reported Hardware Performance (`author-benchmark`, NVIDIA R200 `sm_107a`, 208 SMs, 120 MiB L2, CUDA 13.4)**:
  - Geometric-mean speedup across 74 configurations is **`1.0060x`** on Board A and **`1.0062x`** on Board B.
  - Small-batch rows (`B=1..2`) gain up to **`1.047x`**: `H448-Q1-B1-KV64K` drops from `25.22 / 26.22 us` to **`24.19 / 25.04 us`** (**`1.043x / 1.047x`**); `H192-Q1-B2-KV64K` drops from `25.01 / 26.06 us` to **`23.96 / 24.96 us`** (**`1.044x / 1.044x`**).
  - The routed `H96-Q8-B64-KV64K` configuration improves from `1109.24 / 1102.01 us` to **`1094.48 / 1092.17 us`** (**`1.013x / 1.009x`**).

### 2.3 `cake_mok` Owner-Rank Load Bins & Early Backward Scratch Release (`1e47a692ef2a`, `#7326`)

Auditing `flashinfer/experimental/cake_mok_bf16/{_kernels.py,backend.py,README.md}`:

- **Owner-Rank Load Binning in Count Kernels**: Updates the 9 scheduler count kernels and 1 expert-layout kernel across `sm_100a`/`sm_103a`/`sm_107a` (10 units per arch out of 48 libraries) to bin routed token load per owning EP rank during counting.
- **Precomputed `MoKSchedule.layout` & Single Counter Slab Zeroing**: `build_schedule()` allocates `layout = torch.empty(3 * local_experts + (schedule_capacity // 256), dtype=torch.int32)` and populates it once via `_launch_expert_layout`, while each launch zero-fills only a single counter slab.
- **Immediate Dead Scratch Release in Backward**: Immediately after enqueueing the fused backward kernel (before the output epilogue and `dscores` copy allocate), `backward()` releases `schedule.routed_inter`, `schedule.routed_act`, `schedule.shared_inter`, `schedule.shared_act`, and `schedule.ctx`.
- **Author-Reported Hardware Performance (`author-benchmark`, B200 `sm_100a` & GB300 `sm_103a`)**: On B200 EP8, `glm52_ep8_mxfp8_full_ragged_defaults` step time drops from `15.975 ms` to **`15.596 ms`** (`-2.38%`, **`1.024x` speedup**); on GB300 EP4, `glm52_ep4_mxfp8_full_ragged_defaults` drops from `13.091 ms` to **`12.917 ms`** (**`1.013x` speedup**).

### 2.4 `cake_selective_state_update` Across 122 Shapes (`882d4a6d65e2`, `#7319`) & `cake_mamba_ssd_combined` 15-bit Mask Folding (`5e5835ed42f0`, `#7349`)

- **`cake_selective_state_update` (`#7319`)**: Accelerates all **122 Mamba2 decode shapes** (Nemotron-H and Granite-4.0-H, BF16/FP32 state, canonical FP32 `dt/D` + `int64` slot index or SGLang raw BF16 `dt/D` + `int32` slot index, `hdim=64/128`, `dstate=128`, single/multi-token speculative decode):
  - **B200 (`sm_100a`)**: **`1.015x–1.253x` speedup** (`granite_h_small_hd64_sglang_raw_b1` `3.808 -> 3.040 us`, **`1.253x`**; `sample_stp_bf16` `44.560 -> 39.440 us`, **`1.130x`**).
  - **GB300 (`sm_103a`)**: **`1.018x–1.248x` speedup** (`sample_stp_softplus` `47.168 -> 37.792 us`, **`1.248x`**; `sample_stp_bf16` `47.048 -> 37.760 us`, **`1.246x`**).
- **`cake_mamba_ssd_combined` (`#7349`)**: Introduces `#if __CUDA_ARCH__ >= 1070` `CAKE_SSD_SMEM_DESC_ADDR_MASK = 0x7FFF` (15-bit `tcgen05` SMEM descriptor start-address mask on SM107 vs 14-bit `0x3FFF` on SM100/SM103), folding 21 mask-only `sm_107a` renders (`147` mask sites total) into unified sources, shrinking `generated/device/` from **93 to 72 files** (`_SM107_MODULES` from **45 to 19 entries**) with 63/63 SASS and symbol-table identity proofs.

---

## 3. Reproducible Verification Summary (`own-offline`)

Executing `python experiments/flashinfer_warp_decode_k256_fc2_rubin_mla_pairpipe_mok_load_bins_ssm_update/run.py --check` yields:

| Subsystem / Metric | Baseline / Before | Optimized / After | Speedup / Delta | Provenance |
|---|---:|---:|---:|---|
| `cake_warp_decode` `h4096_e512_i256_k10` (`T=32`, GB200 read cold L2) | `98.18 us` | `93.46 us` | **`1.0505x`** (`38` routed rows) | `author-benchmark` + `own-offline` |
| `cake_warp_decode` `h4096_e512_i256_k10` (`T=31`, GB300 read cold L2) | `95.57 us` | `91.24 us` | **`1.0475x`** (`47` routed rows) | `author-benchmark` + `own-offline` |
| Rubin MLA NVFP4 `H448-Q1-B1-KV64K` (R200 `sm_107a` Board B) | `26.22 us` (`323,584 B` SMEM) | `25.04 us` (`332,800 B` SMEM) | **`1.047x`** (`+9,216 B` SMEM) | `author-benchmark` + `own-offline` |
| Rubin MLA NVFP4 `H96-Q8-B64-KV64K` (`decode_full_pairpipe`, Board A) | `1109.24 us` | `1094.48 us` | **`1.013x`** | `author-benchmark` + `own-offline` |
| `cake_mok` `glm52_ep8_mxfp8_full_ragged_defaults` (B200 EP8) | `15.975 ms` | `15.596 ms` | **`1.024x`** (`-2.38%`) | `author-benchmark` + `own-offline` |
| `cake_selective_state_update` (122 Mamba2 shapes, B200 / GB300) | `3.808 / 47.168 us` | `3.040 / 37.792 us` | **`1.015x–1.253x` / `1.018x–1.248x`** | `author-benchmark` + `own-offline` |
| `cake_mamba_ssd_combined` device files (`0x3FFF` vs `0x7FFF`) | `93` files (`45` SM107 entries) | `72` files (`19` SM107 entries) | **`-21` duplicate renders** | `implementation` + `own-offline` |

---

## 4. Architect Design Exercises & Falsifiable Questions

1. **SMEM Footprint vs. TMA Epilogue Pipeline Depth Calculation**: On `sm_107a`, doubling each correction warp's epilogue staging buffer from one `4 KB` tile to two `4 KB` tiles increases dynamic shared memory from `323,584 B` to `332,800 B` (`+9,216 B`). Calculate the required SMEM footprint if a variant uses 4 correction warps with double-buffered `4 KB` staging tiles, and verify whether it fits within a single CTA on `sm_107a`.
2. **Falsifiable Hypothesis**: For `h4096_e512_i256_k10` at $T \le 6$ decode tokens, the 1-D linear-grid unthrottled `K256 FC2` kernel under-utilizes its 8-token route tiles and therefore does not outperform the existing narrower-tile kernel under cold-L2 conditions (explaining the $T \ge 7$ cutover on `sm_103a` and $T \ge 9$ cutover on `sm_100a`).
