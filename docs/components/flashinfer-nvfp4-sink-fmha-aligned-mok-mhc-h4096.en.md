# Deep Dive: FlashInfer NVFP4 Attention Sinks, Blackwell Aligned Paged GQA Decode, MoK TMEM Hand-Off, Mega-mHC Hidden-4096, Non-Cooperative Causal Conv1d Backward, and TileLang Signed Int64 Atomics

- **Research Date**: 2026-10-24 (Phase 59 · Priority 68 · Track A)
- **Upstream Evidence Chain**:
  - `flashinfer-ai/flashinfer@5be1442520` (`#6021`, *feat(experimental): support attention sinks with NVFP4 KV cache in prefill and decode*)
  - `flashinfer-ai/flashinfer@8656005c6a` (`#7060`, *perf(cake_fmha): add aligned hd128 single-query paged decode kernels*)
  - `flashinfer-ai/flashinfer@4a35122d9e` (`#7059`, *perf(cake_mok): optimize fused kernel issue paths and TMEM hand-off*)
  - `flashinfer-ai/flashinfer@4b88078d61` (`#7048`, *perf(cake_deepgemm_dense_mqa): refresh 32h kernels and extend 64h KV-split tier*)
  - `flashinfer-ai/flashinfer@afb249ac04` (`#6833`, *perf(gemm): tighten cute-dsl bf16 low-M dispatch*)
  - `flashinfer-ai/flashinfer@9ecd6e31b9` (`#7024`, *feat(cake_deepgemm_mega_mhc): add hidden-4096 bf16 inference kernel*)
  - `flashinfer-ai/flashinfer@608772bd85` (`#7072`, *perf(cake_causal_conv1d): replace cooperative bwd with two-launch reduction*)
  - `tile-ai/tilelang@2573d47079` (`#3307`, *Fix CUDA int64 atomic max/min and atomic load/store support*)
- **Evidence Tier**: `implementation` + `author-benchmark` + `own-offline` (`experiments/flashinfer-nvfp4-sink-fmha-aligned-mok-mhc-h4096/run.py`)

---

## 1. Architectural Motivation & Core Bottlenecks

Across Hopper, Blackwell, and Rubin kernel stacks, Phase 59 Track A resolves six critical bottlenecks:

1. **Missing NVFP4 KV Cache Support in Attention Sinks & JIT Module Collisions (`#6021`)**: Attention-sink wrappers lacked packed NVFP4 (`uint8` E2M1 + `fp8_e4m3fn` block scale) KV cache support, and `gen_batch_attention_sink_module` used a static `"batch_attention_sink"` JIT URI that collided across `head_dim=64` and `head_dim=128` in the same process (`0.6–1.2` relative error).
2. **High Paged GQA Decode Latency on Blackwell for Aligned `head_dim=128` (`#7060`)**: Single-query (`q_len_per_req == 1`) paged GQA decode on `sm_100a`/`sm_103a` lacked dedicated cluster-sized (`8/4/2/1` CTAs per `(request, KV head)`) kernels tailored for `head_dim=128` and `page_size % 16 == 0`.
3. **MoK Issue-Path / TMEM Hand-Off Overhead (`#7059`) & 64-Head Dense MQA KV-Split Ceiling (`#7048`)**: `cake_mok`'s 16 fused BF16/MXFP8 routed-expert kernels had instruction-issue and Tensor Memory (TMEM) hand-off stalls, while `cake_deepgemm_dense_mqa` restricted 64-head whole-grid in-kernel KV splitting to `Q > 1024`.
4. **Over-Routing in CuTe-DSL BF16 Low-M GEMM (`#6833`) & Unfused Hidden-4096 mHC on R200 (`#7024`)**: `mm_bf16(backend="cute-dsl")` routed too many medium-area shapes (`M=4` or `M*N <= 8192`) to the SIMT direct runner with only 1 prefetched B vector per thread, while GLM-5.3-Flash (`hidden=4096, n=4`) executed 4 separate kernels for mHC on `sm_107a`.
5. **Cross-Stream Starvation from Cooperative Causal Conv1d Backward (`#7072`) & Unsigned Cast Bug in TileLang `int64` Atomics (`#3307`)**: `cake_causal_conv1d_train_backward` used whole-device `cudaLaunchCooperativeKernel`, stalling by `20x–108x` whenever another CUDA stream held even 1 resident CTA; meanwhile TileLang cast `int64_t` to `unsigned long long` (`NT1`) before `cuda::atomic_ref::fetch_max/min`, inverting negative vs positive comparisons.

---

## 2. Mechanism Deep Dive & Code Walkthrough

### 2.1 NVFP4 KV Cache Attention Sinks & JIT URI Disambiguation (`#6021`)

- **JIT URI Encoding**: `gen_batch_attention_sink_module` now encodes `dtype_kv`, `head_dim_qk`, `pos_encoding_mode`, and `use_fp16_qk_reduction` into the JIT URI, eliminating cross-`head_dim` module collisions (`0.6–1.2` relative error reduced to `~1.8e-3` on NVFP4 decode/prefill).
- **FA2 Scale-Factor Binding & `sm_scale` Folding**: `csrc/attention_sink_prefill_jit_binding.cu` exposes `maybe_k_cache_sf` and `maybe_v_cache_sf` when `DTypeKV == uint8_t`, while `plan()` folds `q_scale * k_scale` directly into `sm_scale = (1 / sqrt(d)) * q_scale * k_scale` and rejects packed FP4 on FA3 (`sm_90a`).

### 2.2 Cake FMHA Aligned `hd128` Single-Query Paged GQA Decode (`#7060`)

`csrc/cake_fmha/fmha_paged_decode_launch.cu` adds 8 aligned `hd128` `q1` paged decode programs per architecture (`sm_100a` and `sm_103a`, covering cluster sizes `8/4/2/1` across BF16 and FP16) for `q_len_per_req == 1`, `head_dim == 128`, `1 <= group_size <= 16`, and `page_size % 16 == 0`:

| GPU | Batch | KV Heads | Seq Len | Dtype | Prev (us) | New (us) | Speedup |
|---|---:|---:|---:|---|---:|---:|---:|
| **B200** | 1 | 8 | 512 | BF16 / FP16 | 24.97 / 19.49 | 4.87 / 5.13 | **5.13x / 3.80x** |
| **B200** | 1 | 8 | 1024 | BF16 | 27.99 | 5.40 | **5.18x** |
| **B200** | 4 | 8 | 4096 | BF16 | 49.71 | 17.73 | **2.80x** |
| **GB300** | 1 | 8 | 512 | BF16 / FP16 | 46.56 / 35.97 | 4.64 / 4.89 | **10.03x / 7.35x** |
| **GB300** | 1 | 8 | 1024 | BF16 | 49.12 | 5.14 | **9.55x** |
| **GB300** | 4 | 8 | 4096 | BF16 | 72.03 | 16.96 | **4.25x** |
| **GB300** | 2 | 8 | 1153 | BF16 | 51.65 | 6.53 | **7.91x** |

### 2.3 Cake MoK Issue-Path / TMEM (`#7059`) & Dense MQA 64h KV-Split (`#7048`)

- **Cake MoK (`#7059`)**: Optimizes instruction issue paths and TMEM hand-off across all 16 fused BF16/MXFP8 routed-expert variants on `sm_100a`/`sm_103a`/`sm_107a` (144 libraries total), cutting B200 EP8 `glm52_ep8_bf16_full_ragged_defaults` step time from **`20.447 ms` to `19.869 ms` (`1.029x`)**.
- **Dense MQA (`#7048`)**: Refreshes 32-head FP8 routes (`+7 / -4` programs) and extends the 64-head in-kernel KV-split tier to `Q <= 1536` (`K > 2048`), retiring `le1024` routes and reaching **`1.0785x`** peak speedup on GB300 (`221.64 -> 205.51 us` at `Q=1024, K=33803`).

### 2.4 CuTe-DSL BF16 Low-M Re-Gating (`#6833`) & Mega-mHC Hidden-4096 (`#7024`)

- **CuTe-DSL BF16 GEMM (`#6833`)**: Narrows direct-runner candidacy to `M <= 3 or M*N <= 4096`, ensures `>= 2` prefetched B vectors per thread at `M <= 4`, sets `outputs_per_block = 1` when `N*K < 2^21`, and prunes warp (`< 96 KiB`) and cluster (`< 144 KiB`) split-K tactics, cutting B300 latency from **`8.35 -> 5.70 us` (`1.465x`)** at `(2, 1000, 8192)` and **`3.75 -> 2.34 us` (`1.603x`)** at `(1, 128, 4096)`.
- **Mega-mHC Hidden-4096 (`#7024`)**: Adds a fused BF16 `hidden=4096, n=4` mHC inference kernel on `sm_107a` (R200), delivering **`1.869x`** (`17.28 -> 9.25 us` at `T=1`) to **`8.598x`** (`970.10 -> 112.83 us` at `T=8192`) speedup over the 4-kernel baseline.

### 2.5 Non-Cooperative Causal Conv1d Backward (`#7072`) & TileLang Signed `int64` Atomics (`#3307`)

- **Two-Launch Causal Conv1d Backward with PDL (`#7072`)**: Replaces `cudaLaunchCooperativeKernel` with an ordinary persistent main kernel writing `dx` and `[rows, C, 4]` FP32 partial `dweight` sums followed by a PDL fixed-order reduction kernel (`max_abs = 0` determinism). Beside an 8-CTA spinning side stream at `S=250,340, C=512`, latency drops from **`20.009 -> 0.306 ms` (`65.4x`) on B200**, **`18.983 -> 0.277 ms` (`68.5x`) on GB300**, and **`16.217 -> 0.149 ms` (`108.8x`) on R200**.
- **TileLang Signed `int64` Atomics (`#3307`)**: Uses `cuda::atomic_ref<T1, cuda::thread_scope_device>` directly with signed `T1` (`int64_t`) instead of unsigned `NT1`, and preserves destination types in `AtomicStore` for `int64`/`bf16`/`fp16`.

---

## 3. Architect Exercises & Falsifiable Questions

1. **Quantitative Exercise**: Compute the exact workspace size `[rows, C, 4]` in bytes for `cake_causal_conv1d_train_backward` on B200 (`148` SMs, `ctas_per_sm=2, grid=592, C=2048`) (`rows = ceil(592 / 32) = 19`, `bytes = 622,592 B`).
2. **Falsifiable Hypothesis**: Casting signed `int64_t` values to `uint64_t` prior to `atomic_ref::fetch_max` causes every negative operand (`-1`, `-2^40`) to evaluate greater than every non-negative operand (`0`, `+2^40`).
