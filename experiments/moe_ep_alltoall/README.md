# MoE Expert Parallelism (EP) & DeepEP All-to-All Model (`own-offline`)

This CPU-only, standard-library experiment (`experiments/moe_ep_alltoall/run.py`) models three primary-source mechanisms from `deepseek-ai/DeepEP` (`b5a33eefda39485f438dc764695b71ba7b9249b6`):

1. **Token Packet Layout & Shared-Memory Warp Sizing (`token.cuh:10-41`, `dispatch.hpp:75-85`, `combine.hpp:44-58`)**:
   - Exact port of `deep_ep::layout::TokenLayout::get_num_bytes<kWithMBarrier>()` enforcing `kNumTMAAlignmentBytes = 32` and `kNumRDMAAlignmentBytes = 64`.
   - Quantifies why `DeepSeek-V3` (`hidden=7168`, `num_sf_packs=56`, `topk=8`) requires `7,552 B` RDMA / `7,584 B` SMEM per token in Dispatch (FP8 + UE8M0 scale factors + metadata) and `14,400 B` RDMA / `14,432 B` SMEM per token in Combine (BF16), consuming `230,912 B` of Hopper's `232,448 B` SMEM across `16` warps (`8` scale-up + `8` forward).
2. **2-Tier Hierarchical Scale-Out (RDMA) + Scale-Up (NVLink) Deduplication (`hybrid_dispatch.cuh:120-135, 408-452`)**:
   - Models `ptx::deduplicate(stored_dst_scaleout_rank_idx, lane_idx)` which sends at most 1 RDMA packet per target node even when multiple top-$k$ experts reside on different GPUs of that remote node, reducing cross-node RDMA traffic by `1.51x–4.01x`.
3. **Zipfian Expert Skew & Redundant Expert Weight Prefetching (`eplb.cuh:15-64, 91-136`, `prefetch_weights.cuh:25-86`)**:
   - Models `EPLBTask` and `EPWeightIterator` (`16 KiB` TMA chunks over NVLink) under Zipfian routing skew, including a small-batch over-replication counterexample where weight prefetch overhead (`5.835 ms`) exceeds the saved expert compute (`0.420 ms`).

## Commands

```bash
python experiments/moe_ep_alltoall/run.py
python experiments/moe_ep_alltoall/run.py --check
```
