# Deep Dive: FlashInfer GDN Prefix-Cache State Materialization, Cake W8A8 MegaMoE EP16 FC2 Evict-First, and Cake FMHA Uniform-Gated L2 Stream Hints

> **中文版**: [flashinfer-gdn-prefix-state-w8a8-megamoe-evict-fmha-l2-stream.md](flashinfer-gdn-prefix-state-w8a8-megamoe-evict-fmha-l2-stream.md) | **Back to Index**: [README.en.md](../../README.en.md) | **7-Card Benchmark Report**: [ai-infra-phase33-benchmark-cards.html](../reports/ai-infra-phase33-benchmark-cards.html)

## 1. First-Hand Source Matrix (2026-10-20)

| Component | Repository & Commit SHA | Release Date | Key Files & Symbols | Evidence Class |
| :--- | :--- | :--- | :--- | :--- |
| **FlashInfer GDN Prefix-Cache State Materialization** | `flashinfer-ai/flashinfer` [`99406bb9cae63a7154dc3498460f4778f1b8a33b`](https://github.com/flashinfer-ai/flashinfer/commit/99406bb9cae63a7154dc3498460f4778f1b8a33b) (`#5073`) | `2026-10-20` | `flashinfer/gdn_kernels/gdn_prefix_materialize.py::gdn_prefix_materialize`, `tests/gdn/reference_gdn_materialize.py` | `implementation` + `own-offline` |
| **FlashInfer Cake W8A8 MegaMoE EP16 FC2 Evict-First L2 Hint** | `flashinfer-ai/flashinfer` [`60213537ce15b522fa75e4526187f505ec36ff70`](https://github.com/flashinfer-ai/flashinfer/commit/60213537ce15b522fa75e4526187f505ec36ff70) (`#6857`) | `2026-10-20` | `flashinfer/experimental/cake_w8a8_megamoe_ep16/csrc/cake_w8a8_megamoe_ep16/cake_w8a8_megamoe_ep16_fc2_kernel.cu::kernel_cake_w8a8_megamoe_ep16_fc2` | `implementation` + `author-benchmark` + `own-offline` |
| **FlashInfer Cake FMHA Balanced Decode Uniform-Gated L2 Stream Hints** | `flashinfer-ai/flashinfer` [`566812deed79eff36b6c1298be9e9ea2d4f101e9`](https://github.com/flashinfer-ai/flashinfer/commit/566812deed79eff36b6c1298be9e9ea2d4f101e9) (`#6111`) | `2026-10-20` | `csrc/cake_fmha/cuda/decode_balanced_16bit/default.cu`, `csrc/cake_fmha/cuda/decode_balanced_hd64/default.cu` | `implementation` + `own-offline` |

---

## 2. GDN Prefix-Cache State Materialization (`gdn_prefix_materialize`)

### 2.1 Why Asynchronous u-Cache Flushes Cannot Directly Feed Prefix Caches
In Gated DeltaNet (GDN) linear-attention decoding, the u-cache mechanism avoids writing the full SSM state matrix every step. Instead, a checkpoint state $S_0 \in \mathbb{R}^{d_v \times d_k}$ resides in `src_slots` while per-token L2-normalized keys $k_j$, delta-rule corrections $u_j$, and cumulative log-decays $g_j$ accumulate in a compact ring buffer.

However, a prefix cache requires states materialized at exact block boundaries (e.g., every 128 tokens), whereas u-cache flushes land at arbitrary speculative/MTP step boundaries. For example, if a request's last flush landed at token 120 and decoding accepted tokens $121\dots 129$ ($\text{hist\_len} = 9$) into the ring, the scheduler needs the exact state at token 128 written to a separate prefix-cache slot (`dst_slots`) **without mutating `src_slots` or disturbing ongoing decoding**.

### 2.2 Partial Non-Destructive Ring Fold and Persistent-Grid Scheduling
Commit `99406bb9cae6` introduces `gdn_prefix_materialize`, replaying only the first $C = \text{count} \le \text{hist\_len}$ ring entries ($C = 128 - 120 = 8$, omitting token 129) onto $S_0$:

$$S_C = \exp(G_C)\, S_0 + \sum_{j=0}^{C-1} \exp(G_C - g_j)\, (u_j \otimes k_j), \qquad G_C = g_{C-1}$$

Because each $u_j$ is already the exact per-token delta-rule update resolved by the WY transform, truncating the fold to $C$ matches step-by-step recurrence with `0.0` error (whereas replaying all 9 entries introduces a `0.130181` boundary state drift in our deterministic verification).

Furthermore, in fixed-batch serving ($B=256, H_V=64$), only a few requests cross a 128-token boundary on any given step:
- The legacy one-CTA-per-item grid incurred **$32.0\,\mu\text{s}$** of launch/scheduling overhead even when `active=0` (`author-benchmark` on B200).
- `gdn_prefix_materialize` uses a **persistent grid with a device-side `num_active` trip count** (skipping negative slot/count rows), cutting `active=0` overhead at $B=256, H_V=64$ to **$2.5\,\mu\text{s}$** (**$12.8\times$ reduction**).

---

## 3. Cake W8A8 MegaMoE EP16 FC2 Evict-First L2 Hints

### 3.1 Streaming Expert Weight L2 Thrashing at 32 Experts/Rank
The `sm103_mxfp8_mxfp8_bf16_cake` (`cake_w8a8_megamoe_ep16`) backend targets $16 \times$ NVIDIA GB300 (`sm_103a`) in an NVL72 domain with 16 ranks, 32 experts per rank, hidden size $N = 3072$, intermediate size $K = 5120$, and top-8 routing. In `kernel_cake_w8a8_megamoe_ep16_fc2` (SwapAB FC2 GEMM fused with routed output scatter), streaming all 32 experts' MXFP8 weights `A` ($15.0\,\text{MiB}$/expert) and E8M0 block scales `SFA` ($0.46875\,\text{MiB}$/expert) sweeps **$495.0\,\text{MiB}$** per rank through cache, evicting reusable routed activations and peer combine metadata.

### 3.2 `0x12F0000000000000ULL` PTX `L2::cache_hint` Injection
Commit `60213537ce15` replaces standard `tma_4d_gmem2smem` calls on `A` and `SFA` with inline PTX carrying the `0x12F0000000000000ULL` (`evict_first`) L2 cache hint descriptor on every $21{,}504\,\text{B}$ TMA transaction. Across 6 token-per-rank shapes ($128\dots 4096$) on $16 \times$ GB300 (`sm_103a`, CUDA 13.3), full MoE layer forward latency drops to a **geomean ratio of $0.991\times$ ($0.986\times\text{–}0.995\times$)** (`author-benchmark`), while our deterministic L2 residency model shows routed-activation hit rate rising from `15.0%` to `94.0%` (`own-offline`).

---

## 4. Cake FMHA Balanced Decode Uniform-Gated L2 Stream Hints

Commit `566812deed79` (`#6111`) extends the `cake_fmha` on-device load-balanced paged-KV decode kernels on SM100a/SM103a (`decode_balanced_16bit/default.cu` and `decode_balanced_hd64/default.cu`) with per-work-item `plan_flags = work_token_words[base + 11]`:
1. **Uniform Split Tiles (`evict_first`)**: When all items share the same sequence length and tile split factor, split tiles load K/V pages under an `evict_first` runtime cache-policy register.
2. **Whole Long Items V-Page Prefetch ($d=64$ only)**: Unsplit long items prefetch V pages into L2.
3. **Heavy Uniform & Mixed-Batch Gate-Off**: Heavy uniform plans ($d=128$ with $\ge 64$ 256-token pairs/CTA) and heterogeneous-length batches clear `plan_flags = 0` to avoid L2 prefetch contention.

---

## 5. Reproducible Experiment & Architect Guidance

```bash
python experiments/flashinfer_gdn_materialize_megamoe_fc2_fmha_l2/run.py --check
```
