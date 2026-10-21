# Deep Dive: FlashInfer Cake All-Gather Matmul (`SM100a`/`SM103a`), Cake MSA/VSA Hopper (`SM90a`), and Blackwell `tcgen05` TMEM Lane-Quarter & CTA-Pair Stage-Drain Invariants

> **Research Phase**: Phase 46 (`priority: 55`) · **Question ID**: `q-phase46-flashinfer-cake-allgather-hopper-msa-vsa-tmem-race` · **Deterministic Script**: [`experiments/flashinfer_cake_allgather_hopper_msa_vsa_tmem_race/run.py`](../../experiments/flashinfer_cake_allgather_hopper_msa_vsa_tmem_race/run.py)

---

## 1. Executive Summary & First-Hand Upstream Commits

Across NVIDIA Hopper (`sm_90a`) and Blackwell (`sm_100a` B200 / `sm_103a` B300), recent commits in `flashinfer-ai/flashinfer` address three critical hardware-software co-design bottlenecks in the Cake kernel stack:

1. **Fused All-Gather + GEMM (`cake_all_gather_matmul`) CTA-Pair Routing & SM-Push Launch Granularity**: On B200 (`sm_100a`) `world_size=2` at `1024x2048` (`K=8192`), routing tables previously selected single-CTA (`_cg1`) copy-engine kernels instead of the 2-CTA (`cta_group::2`, `_cg2`) twin; on `world_size=4` (`512x2560, K=8192`, Llama-3.1-70B TP4 QKV) and B300 (`sm_103a`) `world_size=8` (`1024x1280, K=8192`, Llama-3.1-70B TP8 QKV), the SM-push route launched only `8` CTAs per remote peer (`24` or `56` total CTAs), under-saturating NVLink push bandwidth ahead of consumer GEMM tiles.
2. **Multi-Stage Sparse Attention (`cake_msa`) & Variable Block-Sparse Attention (`cake_vsa`) Tail Imbalance**: Hopper (`sm_90a`) lacked dedicated precompiled Cake MSA programs and relied on general `fi_sm90` CuTe DSL kernels; Blackwell (`sm_100a`/`sm_103a`) union prefill scheduled query tiles in FIFO order with 1-warp Top-K merge, leaving heavy GQA union tiles at the end of execution.
3. **Blackwell PTX ISA `tcgen05` Tensor Memory (TMEM) Addressing & 2-CTA Multicast Stage-Release Races**:
   - In `cake_dsa_train` backward kernels, 128-thread warpgroups omitted `tmem_lane_base = (tid & 96) << 16` in bits `16..22` of `tcgen05.ld`, causing all 4 warps to read TMEM lanes `0..31` and silently discard 3/4 of the warpgroup's accumulators;
   - In `cake_alpha_moe` NVFP4 epilogues, thread-varying `physical_feature = consumer_warp * 32 + lane` was shifted into bits `16..22` of `tcgen05.ld`, violating the PTX warp-uniform `taddr` requirement;
   - In `cake_minimax_h3` (7 stages) and `cake_dense_projection_gemm` (49 CTA-pair kernels), leader CTA (`cta_rank == 0`) issued `tcgen05.commit.cta_group::2...multicast::cluster` (mask `0b11`) to both CTAs' `mma_done[s]` mbarriers without draining the final `min(STAGES, K_steps)` stages before follower CTA (`cta_rank == 1`) exit.

| Commit | Date | Architectural Mechanism | Measured / Verified Impact |
|---|---|---|---|
| `flashinfer@d9695940c5` (`#6865`) | 2026-10-21 | `cake_all_gather_matmul` SM100a `ws2` `1024x2048` `_cg2` routing; `ws4` `512x2560` 16-CTA/peer SM-push (`48` CTAs) | B200 `ws2` BF16/FP16 K-major **`1.065x`** (`101.9 -> 95.4 us`); `ws4` **`1.086x`** (`127.5 -> 113.7 us`) |
| `flashinfer@9c68e44fd2` (`#6761`) | 2026-10-20 | `cake_all_gather_matmul` SM103a `ws8` `1024x1280` (`K=8192`) 16-CTA/peer SM-push (`112` CTAs) | B300 TP8 QKV slowest-rank median **`324.8 -> 317.1 us`** (`1.012x` Node 1, `1.034x` Node 2) |
| `flashinfer@8a93c9b833` (`#6252`) | 2026-10-21 | `cake_msa` Hopper (`sm_90a`) **89-program** route table across all 4 MSA ops + Top-K | Up to **`3.237x`** (`13.775 -> 4.256 us`) on H100 Top-K and **`2.108x`** on FP8 proxy decode vs `fi_sm90` |
| `flashinfer@7f6f119d85` (`#6725`) | 2026-10-20 | `cake_msa` Blackwell persistent heavy-first GQA-folded union prefill + 8-warp rank-count Top-K | **`3.300x / 2.727x`** (GB300/B200) on BF16 paged `first_topk` prefill; **`1.825x / 1.557x`** on flat union prefill |
| `flashinfer@ddf080d8a5` (`#6877`) | 2026-10-21 | `cake_vsa_sm90` **17-program** Hopper package with 4-TMA-box `V^T` load in `k1` and ragged queue | H100 `ext33` **`1.0515x`** geomean; **`1.2524x`** (`13.344 -> 10.639 us`) at `h1-m512-n8192-k48` |
| `flashinfer@c8be875c7c` (`#6778`) | 2026-10-20 | `cake_dsa_train` warpgroup `tcgen05.ld` adds `tmem_lane_base = (tid & 96) << 16` across 80 load sites | Restores full **128-lane** TMEM coverage (`0..31`, `32..63`, `64..95`, `96..127`) across 4 backward kernels |
| `flashinfer@e4e7510f51` (`#6767`) | 2026-10-20 | `cake_alpha_moe` NVFP4 epilogue replaces thread-varying `physical_feature << 16` with `(consumer_warp * 32) << 16` | Enforces warp-uniform `taddr` across **66** `tcgen05.ld` sites per PTX ISA 9.7.16.10.5.2 |
| `flashinfer@653cce0a6e` (`#6820`) & `2b3cd4f2f9` (`#6763`) | 2026-10-20 | `cake_minimax_h3` and `cake_dense_projection_gemm` (**50** `cta_group::2` kernels) add `producer_tail` `mma_done` drain | Eliminates follower CTA early-exit races against `multicast::cluster` (mask `0b11`) mbarrier arrivals |

---

## 2. Cake All-Gather Matmul: `SM100a` `_cg2` Routing and `16`-CTA/Peer Push Geometry

### 2.1 `SM100a` `ws2` `1024x2048` (`K=8192`) CTA-Pair (`_cg2`) Upgrade

In `flashinfer@d9695940c5` (`#6865`), the `sm_100a` routing table upgrades three `world_size=2` `1024x2048` (`K=8192`) copy-engine entries from `_cg1` to `_cg2`:

| Dtype & Layout | Previous Kernel (`_cg1`) | Upgraded Kernel (`_cg2`) | Node 1 Median (`us`) | Node 2 Median (`us`) |
|---|---|---|---|---|
| `BF16` N-major (`nt`) | `agm_ws2_m2048_n2048_k8192_bf16_nt_sm100a` | `agm_ws2_m2048_n2048_k8192_bf16_nt_sm100a_cg2` | `103.5 -> 98.4` (**`1.035x`**) | `102.4 -> 101.6` (**`1.027x`**) |
| `BF16` K-major (`nn`) | `agm_ws2_m2048_n2048_k8192_bf16_nn_sm100a` | `agm_ws2_m2048_n2048_k8192_bf16_nn_sm100a_cg2` | `101.9 -> 95.4` (**`1.065x`**) | `104.2 -> 99.6` (**`1.051x`**) |
| `FP16` K-major (`nn`) | `agm_ws2_m2048_n2048_k8192_fp16_nn_sm100a` | `agm_ws2_m2048_n2048_k8192_fp16_nn_sm100a_cg2` | `101.8 -> 95.3` (**`1.065x`**) | `104.4 -> 100.2` (**`1.052x`**) |

### 2.2 `16`-CTA/Peer SM-Push Launch for `SM100a` TP4 and `SM103a` TP8

On the SM-push path, `push_1d_pull_flag` launches `(world_size - 1) * ctas_per_peer` CTAs to push local shard slices over NVLink before the consumer GEMM reads them:
- **B200 (`sm_100a`) `ws4` `512x2560, K=8192` (`#6865`)**: Raising `ctas_per_peer` from `8` to `16` doubles total push CTAs from `24` to `48` and halves per-CTA payload from `1 MiB` to `512 KiB`, cutting latency from `127.5 -> 113.7 us` (**`1.086x`**) on BF16 N-major and `128.8 -> 114.2 us` (**`1.082x`**) on FP16 K-major.
- **B300 (`sm_103a`) `ws8` `1024x1280, K=8192` (`#6761`)**: Raising `ctas_per_peer` from `8` to `16` at `M == 1024` doubles total push CTAs across 7 peers from `56` to `112` (`1 MiB` per CTA), reducing slowest-rank median latency from `324.8 -> 317.1 us` (**`1.012x`**) on Node 1 and `314.5 -> 311.2 us` (**`1.034x`**) on Node 2.

---

## 3. Cake MSA (`SM90a`/`SM100a`/`SM103a`) & Cake VSA (`SM90a`) Kernels

### 3.1 Hopper (`SM90a`) 89-Program Cake MSA & 17-Program Cake VSA

- **Cake MSA Hopper (`#6252`)**: Compiles **89** dedicated Hopper (`sm_90a`) programs (`22` sparse-decode, `6` sparse-prefill, `24` proxy-decode, `5` proxy-prefill, `32` top-k). On H100 80GB against `fi_sm90` CuTe DSL, `topk_nvp_forced_h4_tiles512_tq256` improves from `13.775 -> 4.256 us` (**`3.237x`**), `topk_plain_h4_tiles512_tq256` from `14.079 -> 4.832 us` (**`2.914x`**), and `proxy_decode_fp8_b4_s256_h32_k4` from `16.929 -> 8.031 us` (**`2.108x`**).
- **Cake VSA Hopper (`#6877`)**: Provides a self-contained **17-program** `sm_90a` variable block-sparse attention package (`2` persistent + `15` small-problem direct/DSM-cluster/split-KV/split-cluster kernels), splitting `k1`'s `V^T` TMA transfer into four `64x16` boxes to eliminate register spills and routing long-head shapes through a heavy-first ragged queue (**`1.2524x`** at `h1-m512-n8192-k48`, `13.344 -> 10.639 us`).

### 3.2 Blackwell Heavy-First GQA-Folded Persistent Union Prefill (`#6725`)

In `flashinfer@7f6f119d85` (`#6725`), Blackwell (`sm_100a`/`sm_103a`) `cake_msa` introduces:
1. **Heavy-First Persistent Union Queue**: Sorts super-tiles descending by union KV tile count so persistent CTAs drain heavy tiles first (`1.92x` tail-load reduction in our 4-SM model);
2. **GQA-Folded Query Tiles**: Folds `num_q_rows * gqa_ratio <= 64` into a single 64-row MMA tile so shared KV pages are loaded once per GQA group;
3. **8-Warp Rank-Count Streaming Top-K Merge**: Parallelizes the `8 * K <= 256` final merge across all 8 warps (`2.133x / 2.129x` Top-K speedup and **`3.300x / 2.727x`** end-to-end BF16 paged `first_topk` prefill speedup on GB300/B200).

---

## 4. Blackwell PTX `tcgen05` TMEM Lane-Quarter & CTA-Pair Stage-Drain Invariants

1. **Warpgroup TMEM Lane-Quarter Base (`#6778`)**: `tcgen05.ld.sync.aligned.32x32b` addresses TMEM with lane in bits `16..22` and column in bits `0..15`. Adding `uint32_t const tmem_lane_base = (tid & 96u) << 16` across 80 load sites in 4 `cake_dsa_train` backward kernels ensures warps `0..3` read lanes `0..31`, `32..63`, `64..95`, and `96..127` (**128** unique lanes instead of **32**).
2. **Warp-Uniform NVFP4 Epilogue `taddr` (`#6767`)**: Replaces thread-varying `(consumer_warp * 32 + lane) << 16` with warp-uniform `(consumer_warp * 32) << 16` across **66** `tcgen05.ld` sites in `cake_alpha_moe`.
3. **2-CTA (`cta_group::2`) Multicast `mma_done` Tail Drain (`#6820`, `#6763`)**: Appends a `producer_tail` loop in `cake_minimax_h3` (7 stages) and **49** `cake_dense_projection_gemm` kernels (4 stages) to await the final `min(STAGES, K_steps)` `mma_done` stage-release barriers before follower CTA exit.
