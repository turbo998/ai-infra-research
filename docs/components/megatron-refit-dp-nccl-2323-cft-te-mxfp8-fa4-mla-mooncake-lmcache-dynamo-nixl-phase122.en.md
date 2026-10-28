# Megatron-LM Single-DP-Group Refit Load Balancing, NCCL 2.32.3-1 Peak BusBW & CFT Counted Windows, TransformerEngine 128x128 MXFP8 Requantize, FlashAttention-4 SM100 Sparse-MLA 64-Head Tail Bound, Mooncake, LMCache, Dynamo & NIXL (Phase 122)

> **中文版**: [megatron-refit-dp-nccl-2323-cft-te-mxfp8-fa4-mla-mooncake-lmcache-dynamo-nixl-phase122.md](megatron-refit-dp-nccl-2323-cft-te-mxfp8-fa4-mla-mooncake-lmcache-dynamo-nixl-phase122.md)
> **Phase**: Phase 122 (`priority: 131`) · Track C Distributed Training, Collectives, KV Tiering & Telemetry
> **Experiment Script**: [`experiments/megatron_refit_dp_nccl_2323_cft_te_mxfp8_fa4_mla_mooncake_lmcache_dynamo_nixl_phase122/run.py`](../../experiments/megatron_refit_dp_nccl_2323_cft_te_mxfp8_fa4_mla_mooncake_lmcache_dynamo_nixl_phase122/run.py)

---

## 1. Primary Sources & Evidence Classification

| Evidence / Source ID | Repo & PR/Commit | Merge Date | Classification | Core Technical Contribution |
|---|---|---|---|---|
| `src-megatron-lm-resharding-single-dp-group-spread-replicas-pr8502-2026` | [`NVIDIA/Megatron-LM#8502`](https://github.com/NVIDIA/Megatron-LM/pull/8502) (`4a55e6fe`) | 2026-10-28 | `implementation` | Removes the `if len(grouped_by_dp) == 1: return src_meta_list[0]` fast path in `_round_robin_dp()`, spreading `TP=1, PP=1` refit reads across all DP replicas via `src_meta_list[dst_rank % len(src_meta_list)]` (**4.0x** egress reduction at `DP=4`) |
| `src-nccl-v2323-gb300-b200-h100-peak-busbw-53a84c7e-2026` | [`NVIDIA/nccl@53a84c7e`](https://github.com/NVIDIA/nccl/commit/53a84c7ec23abe56e152affce86fae36e47c5ef9) | 2026-09-30 | `author-benchmark` | Publishes NCCL 2.32.3-1 Peak BusBW across GB300 (`NVL4..NVL64`), B200 (`NVL8`), and H100 (`NVL8`): GB300 `NVL64` Symmetric (`2GB`) AllReduce reaches **992 GB/s** (**+12.98%** over Legacy `878 GB/s`), and GB300 `NVL8` CE (`8GB`) AllGather/Alltoall reaches **751 GB/s** |
| `src-nccl-cft-counted-operations-symmetric-window-3a87143f-2026` | [`NVIDIA/nccl@3a87143f`](https://github.com/NVIDIA/nccl/commit/3a87143fee5d132217e3e1dd8b04c54dfd08b984) | 2026-09-28 | `specification` + `implementation` | Documents `NCCL_WIN_CFT_COUNTED` and `putCounted` / `putMultimemCounted` / `redCounted`, replacing `data + fence + flag` with hardware-ordered byte-counter increments (`256B` counter alignment, `16B` transfer alignment, $\le 32$ active counters/GPU, $\le 256\text{ GB}$ window) |
| `src-te-grouped-mxfp8-requantize-128x128-persistent-tiles-pr3656-2026` | [`NVIDIA/TransformerEngine#3656`](https://github.com/NVIDIA/TransformerEngine/pull/3656) (`15dcc848`) | 2026-10-09 | `implementation` | Adds a `128x128` wide-tile persistent grouped MXFP8 requantization kernel (`128B` coalesced row segments, `512B` contiguous swizzled scale blocks) that stops at the first tile past `live_rows` (**32.0x** fewer tile evaluations at `4,096 / 32,768` live rows) |
| `src-fa4-sm100-sparse-mla-bwd-preprocess-lse-log2-tail-bound-2865a891-2026` | [`Dao-AILab/flash-attention@2865a891`](https://github.com/Dao-AILab/flash-attention/commit/2865a891a90c01f21d9d1390c063b4fbf2331fd1) | 2026-09-23 | `implementation` | Stops SM100 Sparse-MLA backward preprocess `lse_log2` stores at `seqlen_q` when `use_padded_offsets=False`, preventing a 64-head odd-token sequence's `64`-row `+inf` tail from overwriting the next sequence's first token ($P = 2^{S - \infty} = 0$) |
| `src-mooncake-conductor-configurable-max-blocks-per-context-pr4907-2026` | [`kvcache-ai/Mooncake#4907`](https://github.com/kvcache-ai/Mooncake/pull/4907) (`179ffb21`) | 2026-10-28 | `implementation` | Adds configurable `max_blocks_per_context` to Mooncake Conductor (default `200,000`, batch eviction down to `90%` = `180,000`; `0` disables eviction; negative/non-integer exits with code 1) |
| `src-mooncake-ascend-hccs-2mib-register-alignment-check-pr4903-2026` | [`kvcache-ai/Mooncake#4903`](https://github.com/kvcache-ai/Mooncake/pull/4903) (`e70c0998`) | 2026-10-28 | `implementation` | Rejects non-`2 MiB`-aligned (`2,097,152 B`) remote-accessible `MEM_DEVICE` registrations immediately in `registerLocalMemory` under Ascend Direct `HCCS` mode (`ERR_INVALID_ARGUMENT`) |
| `src-mooncake-pg-apply-view-update-atomic-active-ranks-pr5008-2026` | [`kvcache-ai/Mooncake#5008`](https://github.com/kvcache-ai/Mooncake/pull/5008) (`5ef59b0c`) | 2026-10-28 | `implementation` | Stages `next_active_ranks[64]` on the stack in `MooncakeCommunicator::applyViewUpdate` before writing `meta_->activeRanks` in one pass, eliminating the clear-then-refill false-inactive race window |
| `src-lmcache-mp-grpc-aio-stream-read-ahead-semaphore-pr5858-2026` | [`LMCache/LMCache#5858`](https://github.com/LMCache/LMCache/pull/5858) (`d181fafa`) | 2026-10-28 | `implementation` + `author-benchmark` | Bounds `grpc.aio` server read-ahead with `_STREAM_READ_AHEAD = 1` semaphore, cutting slow-subscriber in-flight batches from `998 -> 158` and server RSS growth from `128 MB -> 4 MB` (**32.0x** reduction) while boosting 1-record batch throughput **2.723x** (`794.5 -> 2163.5 batches/s`) |
| `src-dynamo-sglang-shared-build-postprocessor-streaming-parity-pr17440-2026` | [`ai-dynamo/dynamo#17440`](https://github.com/ai-dynamo/dynamo/pull/17440) (`b7bf2cc4`) | 2026-10-28 | `implementation` | Extracts `_build_postprocessor` in Dynamo's SGLang worker to unify `15` streaming postprocessor settings across inline (`--dyn-preprocess-workers 0`) and worker-pool execution paths |
| `src-nixl-telemetry-prometheus-mp-collision-only-pid-labels-pr2387-2026` | [`ai-dynamo/nixl#2387`](https://github.com/ai-dynamo/nixl/pull/2387) (`250581c1`) | 2026-10-12 | `implementation` | Emits `pid` and `agent_instance` labels in `prometheus_mp` only when $\ge 2$ live writers collide on `(hostname, agent_name, local_rank)`, allowing same-name restarts to continue the existing Prometheus series |

---

## 2. Core Architectural Analysis & Derivations

### 2.1 Megatron-LM Single-DP-Group Refit Balancing (`#8502`) & NCCL 2.32.3-1 Peak BusBW (`53a84c7e`, `3a87143f`)

- **Single-DP-Group Refit Source Selection (`#8502`)**: Under `TP=1, PP=1`, all $D$ parameter replicas share the same `data_parallel_group_ranks` tuple (`len(grouped_by_dp) == 1`). Removing the early `return src_meta_list[0]` fast path reduces selection to:
  $$\text{src\_replica}(\text{dst\_rank}) = \text{src\_meta\_list}\big[\text{dst\_rank} \bmod D\big]$$
  cutting source rank 0 egress load by **4.0x** at $D=4$.
- **NCCL 2.32.3-1 GB300 / B200 / H100 Peak BusBW (`53a84c7e`)**: On GB300, Symmetric (`2GB`) AllReduce scales from `673 GB/s` (`NVL4`) to **992 GB/s** (`NVL64`, **+12.98%** over Legacy `8GB` `878 GB/s`), while CE (`8GB`) AllGather/Alltoall reaches **751 GB/s** on `NVL8`.

### 2.2 TE `128x128` Persistent MXFP8 Requantize (`#3656`) & FA4 Sparse-MLA 64-Head Tail Bound (`2865a891`)

- **TE `128x128` Persistent Grid (`#3656`)**: Expands each CTA's tile from `32x128` to `128x128` with contiguous `512 B` swizzled scale blocks and halts the persistent grid at the first row-block past `live_rows`, avoiding dead CTAs over unpopulated MoE buffer capacity.
- **FA4 Sparse-MLA 64-Head `lse_log2` Tail Bound (`2865a891`)**: Because the backward preprocess tile covers `128` packed `(token, head)` rows, a 64-head odd-token sequence leaves a `64`-row tail that previously wrote `lse_log2 = +inf` over the next sequence's first token when `use_padded_offsets=False` ($P = 2^{S - \infty} = 0$). Stopping `lse_log2` stores at `seqlen_q` eliminates the cross-sequence corruption.
