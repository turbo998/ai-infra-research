# Deep Dive: Megatron-LM Per-Token NVFP4 CuTe-DSL MoE, MFSDPv2 Phase-Selective MXFP8 Plane All-Gather & CUDA-Graph Padding Zeroing, TorchTitan Flux Patch Schedule, LMCache Bigtable L2 Guard, Mooncake Intel XPU P2P & NoF SPDK Decoupling, and NIXL 256B-Aligned SPSC Ring Buffer

- **Research Date**: 2026-10-24 (Phase 57, Priority 66)
- **Evidence Tier**: `implementation` + `author-benchmark` (source audit) + `own-offline` (deterministic stdlib verification)
- **Upstream Commits**:
  - `NVIDIA/Megatron-LM@9314b010da3cd532c7087be4ef7475462cea4e16` (PR `#8262`, 2026-04-20)
  - `NVIDIA/Megatron-LM@18bc49d2f6bc634264c1eef3e2787541d90c6f6c` (PR `#7907`, 2026-04-20)
  - `NVIDIA/Megatron-LM@f9d9bdcc6e09ab432df12394059c8101a33a4440` (PR `#8274`, 2026-04-20)
  - `NVIDIA/TransformerEngine@a7b6c0cbd27fdbf8f1a8d9ed6a9a1d9d012853ba` / `f3826223b7` (PR `#2916`, `#3039`, 2026-04-20)
  - `pytorch/torchtitan@d681f48b43ad348053063d61c1b84b831e3588fe` / `3521b50957` (PR `#5580`, `#5561`, 2026-04-20)
  - `LMCache/LMCache@a3b1ebd9cd5292d1904ed48e459f3ce122e83974` / `8510ceb4be` (PR `#5471`, `#5927`, 2026-04-20)
  - `kvcache-ai/Mooncake@db4f146ecb5ae3bc432c0262c1cb6650a0a39ba4` / `0afc417657` / `a749680052` (PR `#4379`, `#3424`, `#4720`, 2026-04-20)
  - `ai-dynamo/nixl@a9f1f7a76f46f0ee78ec7448936d48ca52073fe3` / `0b146a82d8` (PR `#2018`, `#2483`, 2026-04-20)
  - `ai-dynamo/dynamo@75d0686fb7def243ab5a07ec69159427b169675c` / `f65f04c4fe` (PR `#16765`, `#16534`, 2026-04-20)
- **Verification Script**: `experiments/megatron-nvfp4-moe-mfsdp-planes-pad-zero-mooncake-xpu-nof-nixl/run.py`

---

## 1. Architectural Motivation & Problem Statement

Across distributed training (Megatron-LM, TransformerEngine, TorchTitan) and KV/transfer runtimes (LMCache, Mooncake, NIXL, Dynamo), 17 commits merged on 2026-04-20 solve six systems bottlenecks:

1. **Megatron-LM Per-Token NVFP4 MoE Inference & Batch-Invariant Reduction on GB300 (`#8262`)**: Grouped GEMM MXFP8 MoE suffered from multi-kernel launch overhead and non-deterministic atomic-add finalize under batch-invariant mode.
2. **MFSDPv2 Redundant Dual-Plane MXFP8 All-Gather (`#7907`)**: `QuantizedDBuffer` previously gathered both `rowwise` and `columnwise` MXFP8 weight/scale planes on every `unshard()`, doubling interconnect traffic.
3. **Megatron-LM CUDA-Graph Frozen `padding_slice` Corrupting FP8/FP4 `amax` (`#8274`)**: Full-batch graph capture froze `padding_slice = slice(64, 64)` as a no-op, leaving stale padding activations during partial-batch replay and inducing `2.4e-2` log-prob drift.
4. **TorchTitan Flux `4x` Over-Shifted Timestep Schedule (`#5580`) & Default Regional Compile (`#5561`)**.
5. **LMCache Bigtable L2 Silent Data Loss on Oversized Values (`#5471`) and Mooncake Intel XPU P2P/IPC (`#4379`) & SPDK Decoupling (`#3424`)**.
6. **NIXL SPSC Shared-Memory Ring Buffer False Sharing (`#2018`)**.

---

## 2. Implementation Mechanics & Quantitative Analysis

### 2.1 Megatron-LM: Per-Token NVFP4 CuTe-DSL MoE (`#8262`), MFSDPv2 Phase-Selective MXFP8 (`#7907`), and CUDA-Graph Padding Zeroing (`#8274`)

- **Per-Token NVFP4 CuTe-DSL MoE (`#8262`)**: Uses FlashInfer `0.7.0` CuTe-DSL (`squared_relu` and `swiglu`) with per-token dynamic `w4a4` NVFP4 (`S_global = 1 / (448 * 6)` with `1x16` E4M3 block scales), offline 64-row interleaved `[up, gate]` weight layout, and deterministic two-stage finalize (`use_fused_finalize=False` + `ordered_reduce_scatter_v`) for batch invariance.

| Tokens (GB300 Single MoE Layer `H=7168, I=2048, E=256, top_k=8`) | Default BF16 ($\mu\text{s}$) | Default MXFP8 ($\mu\text{s}$) | Default NVFP4 ($\mu\text{s}$) | Default Speedup (vs BF16 / MXFP8) | Batch-Invariant BF16 ($\mu\text{s}$) | Batch-Invariant NVFP4 ($\mu\text{s}$) | BI Speedup |
|---:|---:|---:|---:|---|---:|---:|---:|
| `1` | `140.4` | `138.2` | **`56.6`** | `2.48x` / `2.44x` | `179.7` | **`82.3`** | `2.18x` |
| `8` | `507.1` | `526.4` | **`187.2`** | `2.71x` / `2.81x` | `575.3` | **`215.2`** | `2.67x` |
| `64` | `1878.8` | `2074.0` | **`626.9`** | `3.00x` / `3.31x` | `1990.7` | **`662.7`** | `3.00x` |
| `512` | `2336.9` | `2433.8` | **`739.0`** | `3.16x` / `3.29x` | `2498.8` | **`772.5`** | `3.23x` |
| `4096` | `4054.7` | `3474.3` | **`1015.6`** | **`3.99x` / `3.42x`** | `5006.8` | **`1107.4`** | **`4.52x`** |

- **MFSDPv2 Phase-Specific MXFP8 Plane All-Gather (`#7907`)**: Gathers only `rowwise` MXFP8 data+scales in `FORWARD` and only `columnwise` MXFP8 data+scales in `BACKWARD`, cutting per-unshard MXFP8 all-gather volume by **`50.0%` (`2x`)**.
- **CUDA-Graph Dynamic Padding Zeroing (`#8274`)**: Triton kernel `zero_padding_rows` reads `gpu_view.real_token_count` (`int32[1]`) from fixed device memory during graph replay (`3.3 us` vs `54.0 us` `masked_fill_` at 2048 tokens on RTX A6000 — **`16.36x` faster**), eliminating stale-padding `amax` corruption.

### 2.2 NIXL `TELEMETRY_VERSION = 6` 256B-Aligned SPSC Ring Buffer (`#2018`) & Backend Phase Trace (`#2483`)

- Aligns `write_pos` (`[0, 8)`) and `read_pos` (`[256, 264)`) to `DESTRUCTIVE_INTERFERENCE_SIZE = 256` bytes (`sizeof(BufferHeader) == 512`, `version` fixed at offset `16`) and caches opposing positions (`cached_read_pos_` / `cached_write_pos_`) with `flock` producer recovery:
  - **Same-Socket**: **`11.9 -> 51.9 M events/s` (`4.36x`)**, RTT **`1022 -> 555 ns` (`1.84x`)**
  - **Cross-Socket**: **`2.5 -> 9.2 M events/s` (`3.68x`)**, RTT **`4448 -> 3353 ns` (`1.33x`)**

### 2.3 Mooncake XPU P2P & NoF (`#4379`, `#3424`), LMCache (`#5471`, `#5927`), TorchTitan (`#5580`, `#5561`), and Dynamo (`#16765`)

- **Mooncake (`#4379`, `#3424`)**: Adds intra-node Intel XPU Level Zero P2P (`queue::memcpy` / `pidfd_getfd` IPC with `8 MiB` host bounce fallback) and decouples SPDK behind `NVMeoFInitiator` + `DmaBufferAllocator` pimpl interfaces.
- **LMCache (`#5471`, `#5927`)**: Returns `False` when any key in a Bigtable L2 batch exceeds `90 MB` (while still writing valid keys) so buffer-only L1 retains the oversized entry, and isolates `close()` steps into independent `try/except` blocks.
- **TorchTitan (`#5580`, `#5561`)**: Fixes Flux schedule shift to use packed `2x2` patch length `(H/2)*(W/2)` instead of `H*W`, and enables default compilation for DeepSeek-V4 `rope` (**`40%`** kernel speedup, **`+1.8%`** prefill) and Muse Spark `fused_binary_activation` (**`+6.3..6.9%`** prefill).
- **Dynamo (`#16765`)**: Eliminates per-step `64 KiB` `routed_experts` cloning in native SGLang response relay.

---

## 3. Architect Exercises & Falsifiable Questions

1. **Communication Exercise**: For a 100B-parameter model under MFSDPv2 MXFP8 ($33/32\text{ B}$ per plane), how much all-gather traffic per rank is saved per training step (1 forward + 1 non-recompute backward)?
   - **Answer**: `192.09 GiB` saved per rank per step (`384.17 GiB -> 192.09 GiB`, `-50%`).
2. **Falsifiable Question**: Does shrinking `DESTRUCTIVE_INTERFERENCE_SIZE` from `256B` to `64B` degrade NIXL cross-socket throughput on x86 CPUs with adjacent cache-line prefetching enabled?
