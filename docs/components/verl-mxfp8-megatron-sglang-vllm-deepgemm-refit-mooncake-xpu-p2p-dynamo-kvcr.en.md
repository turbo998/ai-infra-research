# veRL Blackwell MXFP8 Megatron Training & SGLang/vLLM DeepGEMM Weight Refit, Raw Log-Prob 1-ULP Mismatch Gate, VeOmni Low-Precision ReduceScatter, and Mooncake Intel XPU PCIe P2P + Level Zero IPC & Dynamo KVCR

## 1. Engineering Problem Definition & Cross-Stack Challenges

In Blackwell (`SM100`) RL post-training (GRPO/PPO) and heterogeneous KV cache transfer engines, numerical drift between trainer and rollout engines (`k3_kl`) and intra-node cross-process device copy overhead represent primary architectural bottlenecks:

1. **THD Packed-Sequence Padding Gap in Megatron MXFP8 Training** (`volcengine/verl@d1390eea0c94`, `#8127`): TransformerEngine's MXFP8 quantizer computes `wgrad` block scales along the token dimension in 32-element blocks, requiring both GEMM dimensions to be divisible by 32; legacy THD preprocessing padded sequences only to `lcm(16, align_size)`.
2. **CUDA-Graph Storage Preservation & Kernel Divergence During Bucketed MXFP8/MXFP4/NVFP4 Weight Refit** (`volcengine/verl@5265d561a3bf` `#8306`, `619c397dc991` `#8308`): Rebuilding kernel layouts mid-stream or replacing tensor storage invalidates captured CUDA Graphs, while backend mismatches or unintended Hopper `E8M0` rounding of FP32 block scales inflate trainer–rollout KL divergence.
3. **Float32 1-ULP Blindness in `exp(log_prob)` Mismatch Metrics** (`volcengine/verl@16ec093cc4a2`, `#8025`): Comparing `exp(log_prob)` misses 1-ULP float32 differences (`-0.10000000149` vs. `-0.10000000894` exponentiate to the exact same float32 probability `0.90483743`), breaking strict 0-diff verification gates.
4. **Host-Bounce & Stale Handle Traps in Intra-Node Intel XPU Transfers** (`kvcache-ai/Mooncake@db4f146ecb5a`, `#4379`): Same-node XPU-to-XPU transfers previously bounced through host DRAM or network loopback, and Level Zero's opaque IPC handle cache never invalidates stale mappings after exporter reallocation.
5. **Dynamo KVCR Recurrent Event Quarantine & Per-Token Atomic Counter Contention** (`ai-dynamo/dynamo@8bf042187a8f` `#15535`, `906e89d3a304` `#17067`).

---

## 2. First-Hand Source Code & Empirical Audit

### 2.1 veRL Megatron MXFP8 Recipe (`#8127`) & SGLang/vLLM DeepGEMM Refit (`#8306`, `#8308`)

- **THD Padding Formula (`verl/models/mcore/util.py`)**:
  - With `align_size = tp_size * cp_size * 2` (`cp_size > 1`) or `tp_size` (`cp_size == 1`), `_compute_fp8_thd_align_size(align_size, fp8_recipe)` delegates to Megatron-Core's `get_fp8_align_size(fp8_recipe)`, padding each sequence to **`lcm(32, align_size)`** under `mxfp8` (vs. `lcm(16, align_size)` for other FP8 recipes) and total packed length to `align_size * 128`.
  - Explicitly rejects combining `qat.enable=True` with `override_transformer_config.fp8`.
- **Author-Reported B200 Benchmarks (`author-benchmark`, `#8127`)**:
  - **Qwen3-30B-A3B (`4×B200`, DAPO-Math-17k, 10-step mean)**: `MXFP8/MXFP8` cuts step time from `506 s` (`BF16/BF16`) to **`452 s`** (**`1.1195x`** speedup; generation drops **`23.88%`** from `134 s` to **`102 s`**, `update_actor` drops from `244 s` to **`221 s`**) with `rollout_corr/kl = 0.0045` (vs. `0.0058` for `FP8-TW/BF16` and `0.0090` for `FP8-BW/FP8-BW`).
  - **Qwen3-8B (`2×B200`, DAPO-Math-17k, 32 steps, FP8 KV Cache)**: `MXFP8/MXFP8` matches `BF16/BF16` at **`rollout_corr/kl = 0.0088`**, whereas `FP8-TW/BF16` drifts to `0.0297` with entropy collapse (`0.307 -> 0.062`).
- **SGLang & vLLM Bucket-Staged Refit & DeepGEMM Policy (`#8306`, `#8308`)**:
  - SGLang stages MXFP8 scales and MXFP4/NVFP4 expert tensors across all sync buckets and reconstructs kernel layouts in-place after the final bucket without replacing CUDA-graph-captured storage, while rejecting `delta_flush` and `linear=MXFP8, moe=MXFP4`.
  - vLLM defaults non-NVFP4 low-precision rollouts to `linear_backend=deep_gemm` and `moe_backend=deep_gemm`, and sets `VLLM_USE_DEEP_GEMM_E8M0=0` when Hopper FP32 block scales are detected.
  - **Blackwell Qwen3.5-35B-A3B (`EP=8, TP=2`) `k3_kl` Results**: Matching QAT reduces `BF16 Linear + MXFP4 MoE (DeepGEMM)` `k3_kl` from `0.00571` to **`0.00200`** (**`2.855x`**) on SGLang and from `0.00790` to **`0.00140`** (**`5.643x`**) on vLLM.

### 2.2 veRL Raw Log-Prob 1-ULP Gate (`#8025`) & VeOmni Low-Precision ReduceScatter (`#8330`)

- **Raw Log-Prob Mismatch (`verl/utils/debug/metrics.py`)**: Exposes `training/rollout_logprobs_mismatch_count` (`actor_old_log_probs != rollout_old_log_probs`, aggregated by `sum` in V1 trainer) alongside `rollout_logprobs_diff_max/mean`, catching 1-ULP float32 differences (`7.45e-9`) where `exp(log_prob)` difference is `0.0`.
- **VeOmni Low-Precision ReduceScatter (`#8330`)**: Wires `low_precision_reduce_scatter_comm` (node-local BF16 gradient transport + FP32 accumulation). On `2×H200 NVL` with a 428M dense Qwen3 model, median step times were `117.023 / 118.891 ms` (native FP32) vs. `130.853 / 119.163 ms` (low-precision BF16) with higher reserved memory (`6.416 GiB` vs. `6.318 GiB`), justifying its default-off (`False`) setting for small dense models.

### 2.3 Mooncake TENT Intel XPU PCIe P2P + Level Zero IPC (`#4379`) & Dynamo Optimizations (`#15535`, `#17067`)

- **Mooncake Intel XPU P2P & `pidfd_getfd(2)` IPC (`xpu_sycl_backend.h`, `xpu_transport.cpp`)**:
  - **Intra-Process D2D**: Filters duplicate OpenCL views, probes `sycl_ext_oneapi_peer_access` across ordered Level Zero device pairs, and issues direct `queue::memcpy` when peer access exists (or 8 MB host-bounce chunks otherwise).
  - **Cross-Process Level Zero IPC**: Publishes `"<pid>-<base>-<size>-<handle hex>-<pidns>-<owner uuid hex>-<generation>"`, verifies shared `/proc/self/ns/pid` and peer access to the owning device UUID, fetches the exporter's `dma-buf` fd via **`pidfd_open` + `pidfd_getfd(2)`** to bypass Level Zero's stale opaque-handle cache, and invalidates mappings via monotonic `generation` numbers.
- **Dynamo KVCR Filter Order (`#15535`) & Batched Token Counter (`#17067`)**:
  - Filters `NonMainAttentionKind` / `NonMainAttentionGroup` and allows hash-only `BlockRemoved` (`UnlearnedGroupIdx`) before checking `block_size != kv_block_size`, preventing false KVCR quarantine on hybrid recurrent models.
  - Buffers per-request output token counts in `pending_output_tokens: AtomicU64` (`Relaxed`) and flushes to the shared model counter every **`LOCAL_FLUSH_TOKENS = 64`** tokens or on request completion (**`64x`** reduction in shared cache-line bounces).

---

## 3. Reproducible CPU/Offline Verification

```powershell
python experiments\verl_mxfp8_megatron_sglang_vllm_deepgemm_refit_mooncake_xpu_p2p_dynamo_kvcr\run.py --check
```

| Subsystem / Topic | Configuration | Baseline | Optimized | Key Finding | Evidence Class |
|---|---|---:|---:|---:|---|
| veRL MXFP8 RL (`#8127`) | Qwen3-30B-A3B, `4×B200` DAPO | `506 s/step` (`BF16`) | **`452 s/step`** (`MXFP8`) | **`1.1195x`** step, **`-23.88%`** gen time | `author-benchmark` + `own-offline` |
| veRL DeepGEMM Refit (`#8308`) | Qwen3.5-35B-A3B, vLLM MXFP4 MoE | `k3_kl = 0.00790` (no QAT) | **`k3_kl = 0.00140`** (w/ QAT) | **`5.6429x`** KL mismatch reduction | `author-benchmark` + `own-offline` |
| veRL Raw Log-Prob (`#8025`) | Float32 `-0.10000000149` vs `-0.10000000894` | `probs_diff = 0.0` (missed) | `raw_diff = 7.45e-9` | Captures **1-ULP** diff (`count = 1`) | `implementation` + `own-offline` |
| Mooncake XPU P2P IPC (`#4379`) | Same-node `64 MiB` KV transfer | `3728.27 us` (8MB bounce) | **`1398.10 us`** (P2P IPC) | **`2.6667x`** (`8.0x` vs TCP loopback) | `implementation` + `own-offline` |
| Dynamo Counter Batching (`#17067`) | `4096` output tokens/req | `4096` shared atomic adds | **`64`** shared atomic adds | **`64x`** cache-line bounce reduction | `implementation` + `own-offline` |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Alignment Exercise**: For Megatron THD packing with `tp_size = 2, cp_size = 2`, compute `per_seq_align` and `total_align` under `fp8_block` vs. `mxfp8`.
   - **Solution**: Since `cp_size > 1`, `align_size = 2 * 2 * 2 = 8`. Under `fp8_block`, `per_seq_align = lcm(16, 8) = 16`; under `mxfp8`, `per_seq_align = lcm(32, 8) = 32`; `total_align = 8 * 128 = 1024` tokens for both.
2. **Falsifiable Question**: Why does Mooncake XPU IPC use `pidfd_getfd(2)` plus a monotonic `generation` attribute instead of passing raw Level Zero IPC handles directly?
   - **Verification**: Level Zero's default opaque-handle path caches imported mappings per fd without invalidation; if the exporter frees and reallocates a buffer that reuses the same fd number, the importer maps stale memory unless `pidfd_getfd(2)` + `generation` ordering forces re-import.
