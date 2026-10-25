# FlashAttention-4 1CTA MLA & 3-Stage dV Pipeline, veRL Truncated Score Centering, Megatron Live-Row MoE, Mooncake EFA THP & XPU dma-buf, NIXL GPUNetIO 2-SGE, Dynamo Tiered Prefix & TE Fused MLA KV RoPE MXFP8 (Phase 86)

> **Language / 语言**: [中文](fa4-verl-mcore-nixl-phase86.md) | **English (Current)**

## 1. Background and Cross-Stack Evolution

Across Blackwell (`SM100`/`SM103`/VR200) attention and quantization kernels, RL post-training, deterministic MoE inference, and RDMA/GPUNetIO KV fabrics, Phase 86 Track 3 audits 10 milestone commits from 7 upstream repositories:

1. **FlashAttention-4 (`Dao-AILab/flash-attention`) SM100 1CTA MLA Forward/Backward & `hd256` Semaphore-Free Deterministic Backward** ([PR #2952](https://github.com/Dao-AILab/flash-attention/commit/3451a2a67a24eeed8af54d3c5b8d219577113797), [PR #2938](https://github.com/Dao-AILab/flash-attention/commit/2a92225a91385431ec8b3d095f8d7676d7a48e82)).
2. **TransformerEngine (`NVIDIA/TransformerEngine`) Fused MLA KV Split + Key RoPE + Dual-View MXFP8 Quantization** ([PR #3672](https://github.com/NVIDIA/TransformerEngine/commit/459dedc945a36abbb728479c165acc8a2c196968)).
3. **veRL (`volcengine/verl`) Score Centering over Truncated Sampling Support** ([Commit `08b8387`](https://github.com/volcengine/verl/commit/08b83872283edffb7f710a9bce26bf2a3d6549a7)).
4. **Megatron-LM (`NVIDIA/Megatron-LM`) Batched Packed MTP Context-Parallel Boundary Exchanges & Live-Row Bounded Batch-Invariant MoE** ([PR #8389](https://github.com/NVIDIA/Megatron-LM/commit/50f94464f6ae5fa7f68aac29b6c5da8d3ed16eff), [PR #8385](https://github.com/NVIDIA/Megatron-LM/commit/77232c387ad51e3e0724cf9da9bf50306db98015)).
5. **Mooncake EFA 2MB THP PTE Budget & Intel XPU `dma-buf` GPUDirect RDMA, NIXL GPUNetIO Peer-VRAM & 2-SGE WRITE, and Dynamo Per-Tier Session Prefix Residency** ([Mooncake PR #4564](https://github.com/kvcache-ai/Mooncake/commit/088d599e62b77a4f9a65607dafd2fd4300ed8e2c), [Mooncake PR #4378](https://github.com/kvcache-ai/Mooncake/commit/042ab8a18b0e6ccfe87782261a5556a3dd4446e4), [NIXL PR #2172](https://github.com/ai-dynamo/nixl/commit/14e329032bff3aa3f2431985fad54e91c7da0fed), [Dynamo PR #16501](https://github.com/ai-dynamo/dynamo/commit/c178cbae9b1a2e175a9689a6f8d91d26428675d5)).

---

## 2. First-Hand Source Code & Mathematical Analysis

### 2.1 FlashAttention-4 SM100 1CTA MLA, 3-Stage TMEM `dV` Pipeline & TE Fused MLA KV RoPE MXFP8

- **SM100 `head_dim=256` Dedicated Backward Determinism without Semaphores (PR #2952)**:
  - In `flash_attn/cute/interface.py` and `sm100_hd256_2cta_fmha_backward.py`, every `dQ, dK, dV` tile in the dedicated `hd256` kernel is reduced inside a single CTA in a fixed order without atomics. Dropping `assert not deterministic` and skipping semaphore allocations enables bitwise-deterministic `head_dim=256` training on SM100.
- **SM100 1CTA MLA Forward/Backward Architecture (PR #2938)**:
  - **1CTA `kb64` Mainloop & Wave-Crossover Dispatch (`_mla_1cta_route`)**: Streams **`6.9 TB/s`** of KV payload on GB300 (~86% of HBM3e peak vs `6.2 TB/s` on 2CTA). For unsplit dense decode and sparse top-k gather, 2CTA is kept up to one SM wave on GB300 ($2 \times \text{tokens} \times H_{kv} \le 152$, i.e., $\le 76$ tokens) and switches to 1CTA at $\ge 80$ tokens.
  - **`ptxas -O2` Spill Elimination**: Default `ptxas` spilled **`4064 B/thread`** in `dQdQvGemmKernel`; setting `ptxas_options = "-O2"` reduces spills to **`0 B/thread`** for a **`3.0x–4.2x`** speedup on GB300.
  - **3-Stage TMEM `dV` Accumulator Ring at 64 Heads**: Expands `num_stages_dV` from `2` to `3` ($3 \times 128 + 32 + 32 = 448 \le 512$ TMEM columns, with columns $448..511$ holding fused `dK_rope`), mapping `(group g, split s)` to stage $(2g + s) \bmod 3$ and cutting $T=S=16\text{K}$ main backward time from `18.32 ms` to **`15.29 ms` (1.198x)**.
  - **Padded Recompute-P Backward at $1..64$ Heads**: Zero-fills padded rows via `qheads_first_tma_view` and initializes padded `lse_log2` rows to $+\infty$ so $P = \exp_2(0 - \infty) = 0$ and $\text{dS} = 0$, reducing the 24-head $16\text{K}$-token training step from `30.2 ms` (2CTA load-P) to **`22.5 ms` (1.342x)**.
- **TransformerEngine `nvte_fused_mla_kv_rope_mxfp8` (PR #3672)**:
  - In `fused_mla_kv_rope_mxfp8.cu`, each CTA stages 32 tokens $\times$ 2 heads (`kPitch = 200` BF16 elements), applies Megatron-compatible interleaved-pair RoPE to shared `k_rope (64)` appended after `k_nope (128)`, splits `v (128)`, and quantizes both row-wise (6 warps) and column-wise (3 warps) to compact E8M0-scaled MXFP8 in one pass—cutting `[4096, 1, 128 heads]` time on VR200 from `60.8 + 75.6 = 136.4 us` to **`100.1 us` (1.363x)**.

### 2.2 veRL Truncated-Support Score Centering & Megatron-LM Live-Row MoE

- **veRL `score_centering_support_correction` (Commit `08b8387`)**:
  - When vLLM rollout uses truncated sampling (`top_p < 1` or $0 < \text{top\_k} \le \text{topk\_log\_probs}$), the top-$k$ head holds the entire sampling support $S = \{i : \log q_i > -\infty\}$ with $\sum_{i \in S} q_i = 1$.
  - `score_centering_support_correction` renormalizes the trainer distribution over $S$ ($\log p(S) = \operatorname{logsumexp}_{i \in S} \log p_i$, $p_i^S = p_i / p(S)$) and computes the exact policy-gradient centering term over $S$ while subtracting $\log p(S)$ from the sampled token's log-prob.
- **Megatron-LM Batched MTP CP Exchange & Live-Row Bounded MoE (PR #8389 & PR #8385)**:
  - **PR #8389**: `_exchange_cp_roll_boundaries` batches both adjacent CP boundary transfers into a single `torch.distributed.batch_isend_irecv` call on `cp_group` while preserving repeated padded seam boundaries.
  - **PR #8385**: Bounds batch-invariant MoE permute, activation (`_swiglu_with_probs_kernel`), MXFP8 quantization (`mxfp8_quantize_rows(x, n_used)`), padding zeroing, and FP64-accumulated unpermute by the device-side live row/token count (`offs[-1:]`) rather than worst-case CUDA-graph buffer sizes (`12.8x` work reduction at `1280 / 16384` live rows).

### 2.3 Mooncake EFA 2MB THP / XPU dma-buf, NIXL GPUNetIO 2-SGE & Dynamo Tiered Prefix

- **Mooncake EFA 2MB THP PTE Budget & 3-Stage Fallback (PR #4564)**:
  - Fixes `/proc/self/smaps` VMA header parsing (`"AnonHugePages:"` starting with hex `'A'`), pre-touches `MADV_HUGEPAGE` segments (including tail remainders), and budgets EFA's `22M` PTEs at **2MB** when `Rss == AnonHugePages`—raising the single-NIC registration ceiling from **`88.0 GiB` to `44.0 TiB`** with a 3-tier fallback (`kAuto` $\to$ `kThpPartitioned` $\to$ `kBasePages`).
- **Mooncake Intel XPU Level Zero `dma-buf` GPUDirect RDMA (PR #4378)**:
  - Exports Level Zero device allocations via `ZeLoader` (`libze_loader.so.1`) and registers them with `ibv_reg_dmabuf_mr`, rejecting ambiguous pooled USM allocations via `hasDedicatedDmabuf`.
- **NIXL GPUNetIO Peer-VRAM Registration & Two-SGE WRITE Pairing (PR #2172)**:
  - Registers peer-GPU VRAM under `cudaDeviceGuard(mem.devId)` via PeerDirect `ibv_reg_mr`, and pairs consecutive WRITE descriptors with adjacent remote ranges and matching `rkey` (keeping the final descriptor separate for arrival-marker ordering) into a two-SGE WQE. On dual H20 GPUs ($4096 \times 4\text{ KiB}$ = `16 MiB`), data WQEs drop from `4096` to **`2049`**, median bandwidth rises from `10.397` to **`12.697 GB/s` (+22.12%)**, and Tx p99 drops from `1613` to **`1325 us`**.
- **Dynamo Per-Storage-Tier Session Prefix Residency (PR #16501)**:
  - Keying `SessionPrefixIndexer` frontiers by `(WorkerWithDpRank, StorageTier)` tracks deep prefix residency across GPU and lower offload tiers simultaneously.

---

## 3. Reproducible Verification (`experiments/fa4_verl_mcore_nixl_phase86`)

Running `python experiments/fa4_verl_mcore_nixl_phase86/run.py --check` deterministically verifies all metrics:
- **FA4 & TE**: `4064 -> 0 B` spill elimination, `448` TMEM columns (`1.198x` faster 64h `dV` pipeline), and `1.363x` faster TE fused MLA KV RoPE MXFP8;
- **veRL & Megatron-LM**: `1.000000` renormalized support sum, `0.0` centering residual sum, and `12.8x` live-row MoE work reduction;
- **Mooncake, NIXL & Dynamo**: `88.0 GiB -> 45056.0 GiB` single-NIC EFA limit, `4096 -> 2049` WQEs, and `+22.12%` NIXL GPUNetIO bandwidth gain.

## 4. Architect Exercises & Falsifiable Questions

1. **GPUNetIO WQE Count Derivation**: Why do $4096$ contiguous 4 KiB descriptors produce exactly $2049$ WQEs under `canPairWrite` (`remaining > 2`) instead of $2048$?
2. **Falsifiable Hypothesis**: In veRL under `top_p = 0.9, top_k = 64`, if `score_centering_top_p` is omitted when $p(S) < 0.7$, how does the unnormalized importance ratio $p_a / q_a$ bias policy gradient updates?
