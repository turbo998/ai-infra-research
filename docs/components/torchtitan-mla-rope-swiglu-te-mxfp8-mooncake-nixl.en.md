# TorchTitan In-Row Int32 MLA RoPE & Unmasked SwiGLU Backward, TransformerEngine Per-Tile MXFP8 Scale Swizzle, Megatron Async Payload Staging, and Mooncake/LMCache/NIXL/Dynamo Transport Hardening

> **中文版**: [torchtitan-mla-rope-swiglu-te-mxfp8-mooncake-nixl.md](torchtitan-mla-rope-swiglu-te-mxfp8-mooncake-nixl.md)  
> **Phase 63 · Priority 72 · Track C (`question-204`)** · Verified Date: `2026-10-24` (UTC+08:00)

---

## 1. Core Architectural Problem & Context

Across `torch.compile` regional compilation for DeepSeek-V3 (671B) and Kimi K2.5, Blackwell MXFP8 scale swizzling, dynamic multimodal inference staging, and RDMA/UCX KV cache fabrics, five high-impact bottlenecks surfaced:

1. **TorchTitan Strided MLA RoPE Slicing & Masked SwiGLU Backward Bandwidth Under-Utilization**:
   - In `_apply_rope_to_q_and_k_pe` (`mla_qk` region), splitting $q \in \text{BF16}[T, H, 192]$ into `q_nope [.., 128]` and `q_pe [.., 64]` yields a strided view for `q_pe`. Inductor cannot view a strided slice as packed `int32` pairs without copying, falling back to stride-2 loads and achieving only **59%–69%** of the GB300 HBM bandwidth floor (`1380.0 us` vs `945.7 us` floor on DSv3 64k prefill).
   - In the compiled `SwiGLU` region, Inductor wrote `grad_gate_up [8T, 2, F]` via a single masked (`tl.where`) kernel, taking `788 us` at $T=16\text{k}, F=2048$ (**2.1x** the `377 us` bandwidth floor).
2. **TransformerEngine MXFP8 Narrow-$K$ Scale Swizzle CTA Starvation**: The legacy narrow-$K$ kernel packed up to `32` tiles of `128x4` scales into a single `1024`-thread CTA, launching only **1 CTA** for a `4096`-row activation segment.
3. **Megatron-LM `DynamicInferenceEngine` Synchronous Staging & Residual GPU Memory Retention**: `payload_stager.stage()` blocked the main engine loop during H2D transfers, and finished requests retained GPU-backed `prompt_tokens` and VLM `media_tensors`.
4. **Mooncake Unconditional RDMA Port Polling & TENT `cuda_memcpy_sync` Host `memcpy` Fallback Crash**:
   - Mooncake's 1 Hz RDMA recovery thread previously polled `ibv_query_port` and probed GIDs after **any** context-disabling event (including unrecoverable `IBV_EVENT_DEVICE_FATAL`, `IBV_EVENT_CQ_ERR`, and `IBV_EVENT_WQ_FATAL`).
   - In TENT `cuda_memcpy_sync`, worker threads without an active `CUcontext` failed `cudaMemcpy`, and `DataCopy` treated non-`TENT_DEVICE_NOT_SUPPORT` errors as eligible for a host `std::memcpy` fallback on raw GPU pointers, crashing the process.
5. **LMCache Periodic Thread Races, NIXL UCX Sender Blind Spots, & Dynamo First-Request Cold Start**: Concurrent `PeriodicThread.stop()` calls raced on `join()`, `MooncakeLookupClient` dropped `request_configs` tags during batch lookups, NIXL UCX lacked sender-side phase tracing, and Dynamo lazily initialized NIXL connectors on the first inference request.

---

## 2. Upstream Implementation Paths & Quantitative Mechanics

### 2.1 TorchTitan: In-Row `int32` Pair MLA RoPE (`2fddc6b7a0` / `#5529`, `21db340cba` / `#5161`) & Unmasked SwiGLU (`eb48bab350` / `#5460`)

1. **`_RotateTail` In-Row `int32` Pair Rotation (`#5529`)**:
   - Commit `21db340cba` (`#5161`) first compiles $q$'s split $\to$ RoPE $\to$ cat and `k_pe`'s RoPE as the `mla_qk` region with `use_contiguous_tensor_cat=True`, saving `3.9 ms` (`182.44 -> 178.58 ms`, `-2.1%`) on a 2-layer DSv3 671B 16k step on GB300.
   - Commit `2fddc6b7a0` (`#5529`) then views the contiguous row `q [T, H, 192]` as `int32 [T, H, 96]` via `_RotateTail`, rotates the last `32` `int32` words (`q_pe`'s `64` BF16 dims) in-place with `_RotatePairs.rotate`, and concatenates them with `q_nope`'s `64` untouched words.

| Model & Mode (GB300, CUDA Graphs) | Sequence Length $T$ | Before ($\mu\text{s}$) | `#5529` ($\mu\text{s}$) | Delta / Speedup | HBM Floor ($\mu\text{s}$) | Floor Utilization |
|---|---:|---:|---:|---|---:|---:|
| **DeepSeek-V3 671B** · `train` | `16,384` | `761.3` | **`492.8`** | **-35.3%** (**1.545x**) | `472.8` | **95.9%** |
| **DeepSeek-V3 671B** · `train` | `65,536` | `2985.5` | **`1926.6`** | **-35.5%** (**1.550x**) | `1891.3` | **98.2%** |
| **DeepSeek-V3 671B** · `prefill` | `65,536` | `1380.0` | **`953.5`** | **-30.9%** (**1.447x**) | `945.7` | **99.2%** |
| **DeepSeek-V3 671B** · `prefill` | `131,072` | `2748.1` | **`1899.5`** | **-30.9%** (**1.447x**) | `1891.3` | **99.6%** |
| **Kimi K2.5** · `train` | `65,536` | `1518.6` | **`989.7`** | **-34.8%** (**1.534x**) | `950.6` | **96.0%** |
| **Kimi K2.5** · `prefill` | `131,072` | `1429.8` | **`977.9`** | **-31.6%** (**1.462x**) | `950.6` | **97.2%** |

2. **SwiGLU Unmasked Dual-Half Stores (`eb48bab350` / `#5460`)**: Setting both `max_pointwise_cat_inputs=0` and `max_complex_pointwise_cat_inputs=0` on the SwiGLU compile region emits one unmasked store per half, cutting GB300 $T=16\text{k}, F=2048$ fwd+bwd latency by **27.2%** (`1440.0 -> 1047.8 us`, **1.374x**), outperforming the hand-written Triton override (`1060.5 us`).

### 2.2 TransformerEngine Per-Tile MXFP8 Scale Swizzle (`33e7ff3160` / `#3658`) & Megatron Async Staging (`894b5114e3` / `#8169`)

1. **TE `NVTE_MXFP8_PER_TILE_SWIZZLE=1`**: Assigns 1 CTA (`128` threads row-wise / `256` threads col-wise) per `128x4` scale tile (up to `kRowwiseMaxBlocks = 1536`, `kMaxTensorsPerKernel = 64`), increasing CTA parallelism **32x** (`1 -> 32` CTAs) on a `4096`-row activation. Companion commit `#3540` (`1c9f6f8f39`) unlocks FA4 $D=256$ forward on SM100/SM110 and guards THD `seqused_q/k` across FA4 `b31`/`b33`.
2. **Megatron-LM Background Staging & Zero GPU Retention (`894b5114e3` / `#8169`)**: Offloads `payload_stager.stage()` to a background thread pool (`_staging_tasks`) and detaches `prompt_tokens` and VLM `media_tensors` to CPU in `OffloadedRequestPayload.from_request` so finished requests hold **`0 B`** of GPU memory. Commits `fa322ce620` (`#8010`) and `0156c21373` (`#3698`) fix meta-device tied-embedding materialization before `all_reduce` and `MultiLatentAttention` `mscale * mscale` scaling when `config.softmax_scale` is set.

### 2.3 Mooncake RDMA/TENT Recovery, LMCache Thread Safety, & NIXL/Dynamo Transport Lifecycle

1. **Mooncake RDMA 1 Hz Recovery Gated on `IBV_EVENT_PORT_ERR` (`8c19b7286c` / `#4715`)**: Tracks `context_disabling_event_` and restricts 1 Hz `ibv_query_port` + `probeRdmaContext()` recovery strictly to `IBV_EVENT_PORT_ERR` when the circuit breaker has expired, refusing recovery on `DEVICE_FATAL`, `CQ_ERR`, or `WQ_FATAL`.
2. **Mooncake TENT Context Save/Restore & `TENT_DEVICE_PLUGIN_ERROR = -3` (`b2c7d0a7b5` / `#4839`)**: Retains and binds the primary `CUcontext` when `cuCtxGetCurrent` is null in `cuda_memcpy_sync`, restores the caller's context on exit, and returns `TENT_DEVICE_PLUGIN_ERROR = -3` to prevent `DataCopy` from falling back to host `memcpy` on GPU pointers.
3. **Mooncake Namespace Eviction (`fff8e90952` / `#4811`), LMCache Locks (`e0e92b8356` / `#5281`, `e45d4d5a85` / `#5449`), NIXL UCX Sender Traces (`8d5660c118` / `#2443`), & Dynamo Warmup (`6278e410d2` / `#17060`)**: Separates `Evictor` and `OffloadingReclaimer` with per-namespace watermarks, synchronizes `PeriodicThread.stop()`, forwards `request_configs` in `MooncakeLookupClient.exists`, instruments `nixl::submit` / `nixl::wire.submitted` / `nixl::notif.sent`, and eagerly runs `nixl_connector.warmup()` during Dynamo engine startup.

---

## 3. Deterministic Verification & Architect Guidance

Verified by `experiments/torchtitan-mla-rope-swiglu-te-mxfp8-mooncake-nixl/run.py` and `tests/test_phase63_experiments.py`:
- Enable TorchTitan's `mla_qk` in-row `int32` pair rotation and unmasked SwiGLU options (`max_pointwise_cat_inputs=0, max_complex_pointwise_cat_inputs=0`) on GB300 to reach **95.9%–99.6%** of the HBM bandwidth floor without custom Triton overrides.
- Gate automated RDMA port recovery strictly on `IBV_EVENT_PORT_ERR` and distinguish device-plugin runtime failures (`TENT_DEVICE_PLUGIN_ERROR = -3`) from unsupported-device fallbacks.
