# FlashInfer SM120 MegaMoE Register-Spill Elimination, TRT-LLM-Gen Multi-Wave KV Split, Kimi-K3 AttnRes Round-2 & NVFP4 Attention Package, and TRT-LLM Zero-Copy Video Encoding

> **First-Hand Upstream Sources**:
> - `flashinfer-ai/flashinfer@cd6c1885a897c95a3728aae6eb7a2d14ca88aa16` (`fix(cake_mega_moe): [SM120] MegaMoE register spill with CUDA 13.3 headers; compile-check every EP (#6770)`, 2026-10-21)
> - `flashinfer-ai/flashinfer@a3f2525e39dffc98dcad4ce50706641a580358f0` (`perf(trtllm-gen): opt-in multi-wave KV split for decode when one wave gives <= 2 splits (#6331)`, 2026-10-21)
> - `flashinfer-ai/flashinfer@3b65505f40e34d1bd08d9364811057242e3342fd` (`perf(cake_kimi_k3_attn_res): round-2 Kimi-K3 AttnRes routes and programs for SM100 / SM103 as a vendorable cake-build package (#6904)`, 2026-10-21)
> - `flashinfer-ai/flashinfer@edfb3f7f883950308f74ce6e8d41759eb21afa12` (`refactor(cake_nvfp4_attention): ship the experimental NVFP4 attention as a self-contained vendorable package (#6902)`, 2026-10-21)
> - `NVIDIA/TensorRT-LLM@c018848da6f1250dcf7abb7eaf3b113a774f9aa0` (`[None][fix] Encode VisualGen video without a GIL-held frame copy (#20575)`, 2026-10-20)

---

## 1. SM120 Cake MegaMoE Register-Spill Elimination Under CUDA 13.2–13.4 (`flashinfer@cd6c1885a8`)

### 1.1 Root Cause: CCCL `cuda::atomic_ref` Local-Storage Guard & Multi-GIN Inlining
The `sm120_mxfp4_mxfp8_bf16_cake` persistent MegaMoE kernel runs right at the **168-register ceiling (`SM120_REG_CAP = 168`)** and **8-byte local stack limit (`LOCAL_BYTES_LIMIT = 8`)** (`1 spilled register = 16 bytes local memory`). Any spill causes `_build_workspace` to reject the cubin.
Upstream traced `34/352` spills under CUDA 13.3 and `80/352` under CUDA 13.4 to three sources:
1. **CCCL `isspacep.local` branch bloat (+10 regs)**: Starting in CUDA 13.2, `libcu++` inserts an automatic-storage address-space check (`isspacep.local`) and local-memory fallback around every `cuda::atomic_ref` operation on a generic pointer. NCCL's `nccl_device.h` uses `cuda::atomic_ref` exclusively on NCCL-managed device/host-pinned GIN descriptors, signals, and counters—never thread-local stack variables.
2. **Runtime `switch` across three GIN backends (+10–20 regs & 8 B stack)**: Compiling `PROXY`, `GDAKI`, and `GPI` into a single kernel caused cross-backend register pressure and forced an 8 B stack frame from the GPI barrier even when running proxy/GDAKI.
3. **`m128` 4-step K unroll and >16-rank per-source tables in registers**: `m128` unrolled 4 MMA K steps per loop body, and `EP > 16` (`EP=32`) kept per-source routing tables in register arrays.

### 1.2 Three-Part Fix & 15-Bit Signed ABI Cap (`ACTIVE_ROWS_FIELD_MAX = 0x7FFF`)
- **Part A (`_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE`)**: Defined before `nccl_device.h` and passed via `-D_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE`, eliminating `isspacep.local` branches while producing byte-identical cubins on CUDA 13.0.88 across all 64/64 instances.
- **Part B (Per-GIN modules + `m128` 1-step K loop + `tb_pub` workspace)**:
  - Compiles separate per-GIN modules (`NCCL_GIN_{PROXY,GDAKI,GPI}_ENABLE`), dropping stack frame size to `0 B` on `PROXY` and `GDAKI`;
  - Emits 1 MMA K step per loop iteration in `m128` and moves per-source routing tables into the `tb_pub` shared workspace above 16 ranks;
  - **15-Bit Signed ABI Guard**: `active_rows` is a signed `int32_t` packing per-call active rows in the low 16 bits and per-rank max rows in the high 16 bits decoded via an **arithmetic right shift (`active_rows >> 16`)**. Setting the cap to `0xFFFF` (`65408`) sets bit 31 when `max_rows >= 32768 (0x8000)`, decoding as negative (`-32768`). Tightening `ACTIVE_ROWS_FIELD_MAX` to `0x7FFF` bounds `max_rows <= 32640` (`128`-aligned).
- **Verification**: Across **11 CUDA 13 releases (`13.0.48` through `13.4.92`) × 3 GIN backends × 480 instances (`EP = 4, 8, 16, 32`) = 1,440 JIT instances per toolkit**, spills drop to **0/1440 (`130–168` registers, `0 B` local memory)**, while adding `DeepSeek-V4.1-Flash` (`E384 / H5120 / I2304 / top-k 6`).

---

## 2. TRT-LLM-Gen GQA Decode Opt-In Multi-Wave KV Split (`flashinfer@a3f2525e39`)

### 2.1 Why Single-Wave Splits Starve Skewed Long-KV Decode Batches
By default, `TllmGenFmhaKernel` assigns `splits = SMs // (batch * headsKv)` so total CTAs fit within a single wave. On NVIDIA B200 (`148 SMs`) with 32 Q heads and 2 KV heads (`base_ctas = 2 * batch`):
- At `B = 25..37`, single-wave splitting allows only `2` splits per sequence;
- At `B = 38..148`, single-wave splitting collapses to **`1` split per sequence**, forcing a single CTA pair to serialize the longest KV cache in skewed (`lognormal, sigma=0.6`) decode batches.

### 2.2 `FLASHINFER_TRTLLM_GEN_MULTI_WAVE_KV_SPLIT=1` Up to 4 Waves with Exact Scratch Fallback
Commit `a3f2525e39` introduces an opt-in multi-wave KV split policy (`kMaxNumWavesForKvSplit = 4`) applied strictly to the final selected GQA decode kernel when single-wave splitting yields `splits <= 2`:
- **Split Expansion**: `candidate_splits = (4 * SMs) // base_ctas` (up to `11` splits: `2 -> 11` at `B=25`, `1 -> 7` at `B=38`, `1 -> 4` at `B=74`).
- **Three Hardware & Scratch Invariants**:
  1. `static_assert(kMaxNumWavesForKvSplit <= 4)` and `base_ctas <= 2 * SMs` so the standalone reduction kernel (`fmhaReduction.cu`) never computes a zero-sized grid;
  2. `partialStatsBufferSize = max(mMultiProcessorCount, totalNumCtas) * mStepQ` so the partial-stats scratch scales with actual launched CTAs when `totalNumCtas > SMs`;
  3. **Exact Scratch Fallback**: The launcher passes remaining workspace budget `mMultiCtasKvScratchBytes`; if multi-wave CTAs exceed the budget by even **1 byte** after CGA/GMEM reselection, it falls back cleanly to the single-wave split count.
- **Measured Speedups (B200 148 SMs, CUDA Graph `max_seq_len=1M`)**:
  - **1.86x geomean** speedup across changed points on `64k lognormal` (`B=38`: `311 -> 177 us`, **1.75x**; `B=74`: `721 -> 386 us`, **1.87x**);
  - **2.02x peak** at `B=38` on `288k lognormal` (`1566 -> 775 us`), and up to **3.0x** on SM107 at `B=54..64`.

---

## 3. Cake Kimi-K3 AttnRes Round-2, NVFP4 Attention Package, and TRT-LLM Zero-Copy Video Encoding

| Component & Commit | Mechanism | Verified Metrics & Contracts |
| :--- | :--- | :--- |
| **Cake Kimi-K3 AttnRes Round-2** (`flashinfer@3b65505f40`) | Fuses residual add, optional snapshot write (`block_write_idx == K`), `K+1` candidate scoring, softmax, mixing, and output RMSNorm for `H=7168, K in [0, 8]` BF16 into a `cake-build/1` package (`133` programs, `161` libraries: `75` `sm_100a` + `86` `sm_103a`). | Across `291` contract shapes × 2 PDL modes (`582` rows): **1.0072x** geomean on GB300 (**1.0888x** at `serving_m5_k5`, `4.448 -> 4.096 us`), **1.0064x** on B300, and **1.1034x** peak on B200 (`k_sweep_m1_k6`, `5.120 -> 4.640 us`). |
| **Cake NVFP4 Attention Package** (`flashinfer@edfb3f7f88`) | Refactors SM103a persistent-tile NVFP4 attention (`HEAD_DIM=128`, `SF_VEC=16`, `TILE_ROWS=512`, `E2M1_BOUNDARIES=(0.25..5.0)`) into a self-contained `cake-build/1` package with AOT `.so` loading and JIT fallback. | **0.9975x** geomean parity on GB300 across 4 benchmark shapes (`B=4, H=8, S=4096`: `114.161 us` vs `114.177 us`, **1.000x**). |
| **TRT-LLM VisualGen Zero-Copy Encoding** (`TensorRT-LLM@c018848da6`) | Replaces `video_np.tobytes()` with `memoryview(video_np.reshape(-1))` when streaming raw RGB24 frames into `ffmpeg` stdin in `encode_video`. | Eliminates a **711.91 MiB** (`746,496,000 B`) host buffer allocation and GIL-held full-video copy for 120-frame 1080p RGB video. |

---

## 4. Deterministic Reproduction (`experiments/flashinfer-sm120-megamoe-spill-trtllm-multiwave-kimi-k3-nvfp4/run.py`)

```bash
python experiments/flashinfer-sm120-megamoe-spill-trtllm-multiwave-kimi-k3-nvfp4/run.py --check
```
