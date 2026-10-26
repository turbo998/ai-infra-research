# SGLang Confirmed-First Cold-Prefix Routing & HiCache Mooncake L3 Layout, vLLM RL ParamMeta & Five-Fold KV Offload Hardening, and TensorRT-LLM Mamba/KDA Replay & Beam Hypothesis Gathering (Phase 88)

> **Language / 语言**: [中文 (`sglang-router-hicache-vllm-kv-offload-trtllm-phase88.md`)](sglang-router-hicache-vllm-kv-offload-trtllm-phase88.md) | **English (Current)**

- **Queue ID**: `sglang-router-hicache-vllm-kv-offload-trtllm-phase88` (`priority: 97`)
- **Evidence ID**: `ev-sglang-router-hicache-vllm-kv-offload-trtllm-phase88-2026`
- **Verification Status**: `own-offline` (Primary upstream source audit + deterministic Python 3.12 stdlib verification in [`experiments/sglang_router_hicache_vllm_kv_offload_trtllm_phase88/run.py`](../../experiments/sglang_router_hicache_vllm_kv_offload_trtllm_phase88/run.py))

---

## 1. Executive Summary & Architectural Problem

Across production LLM inference and RL rollout engines, recent 2026-10-26 commits in SGLang, vLLM, and TensorRT-LLM resolve six high-impact failure modes and bottlenecks:
1. **Cold-Prefix Burst Pile-Up in KV-Aware Routing**: When 64 concurrent requests arrive sharing a cold prefix, unbounded pending-prefix affinity funnels all 63 followers onto the single worker holding the first unconfirmed stake.
2. **Opaque Mooncake L3 Key Layouts**: External KV-aware routers could not inspect which Mooncake L3 object keys resided on each SGLang DP rank.
3. **Multi-Batch LRU Prefix Eviction Inversion in vLLM CPU Offloading**: Without absolute token-position metadata (`req_context`), multi-batch LRU eviction could evict earlier prefix blocks of a running request while keeping tail blocks, violating the prefix-continuity contract.
4. **Tiered KV Offload Control/Data Plane Races in vLLM**: Deadlocked remote P2P peers blocked `ZmqConnection.send()` indefinitely; in-flight tiered promotions dirtied newly reset caches; failed NIXL remote registrations leaked partial state; OBJ tier files collided across different `blocks_per_chunk` specs.
5. **TensorRT-LLM Speculative GDN/KDA Replay Mis-Advancement & Blackwell TMA Backing Underrun**: Advancing cached replay state by all draft tokens rather than accepted + bonus tokens corrupted Mamba/KDA state on speculative steps, and sub-128 KiB staging pools faulted on Blackwell native sparse FMHA TMA descriptors.
6. **Beam Search Logits Processor History Corruption**: Passing raw beam slot tables instead of gathered beam hypotheses caused logits processors to penalize or constrain the wrong token sequences.

### Primary Source Matrix

| Upstream Commit / PR | Target Module / Path | Core Mechanism | Quantitative Result |
|---|---|---|---|
| `sgl-project/sglang` `#46505` (`eabb11a4c1`, 2026-10-26) | `sgl-router/src/config/{cli,types}.rs` | Adds `--cache-pending-prefix-consult-below <ratio>` (confirmed-first) and `--cache-pending-prefix-max-followers <cap>` | Cuts 64-request cold-prefix burst max single-worker load across 8 workers (`max_followers=7`) from `64` to `8` (`8.0x` reduction) |
| `sgl-project/sglang` `#40467` (`64fc5f0bdb`, 2026-10-26) | `python/sglang/srt/entrypoints/http_server.py` | Exposes `/get_hicache_l3_cache_layout` merging per-DP-rank Mooncake L3 object keys | Enables external KV-aware routers to route directly to L3-resident workers |
| `sgl-project/sglang` `#33250` (`ef11eb3b76`), `#45713` (`6a28c96a49`), `#46308` (`9841183827`) | `moe_transport.py`, `kda_decode_gfx950.cu`, `ipc_a2a_multi.py` | Fixes DeepSeek-V4 non-EP TBO padding under `attn_tp > 1`; enables fused HIP KDA decode on MI350X/MI355X (`gfx950`); hides Ulysses own-row copy | `gfx950` KDA decode fuses 5 stages into 1 launch (`5.0x` launch reduction); zero extra Ulysses self-copy bubble |
| `vllm-project/vllm` `#63256` (`50a184f435`) & `#61636` (`14d593115d`) | `docs/training/weight_transfer/`, `vllm/lora/model_manager.py` | Declares RL weight ownership & shard layout on `ParamMeta` (`nccl_m2n` avoids `full_tensor()`); fixes 3D MoE LoRA double-pinning & EP dummy LoRA sizing | Cuts 3D MoE LoRA pinned host memory by `2.0x` and aligns EP dummy warmup to global expert count |
| `vllm-project/vllm` `#61050` (`e113018f50`), `#63708` (`51d2a85594`), `#62929` (`14aefcd4f9`), `#63628` (`8a4b018f1c`), `#63655` (`8c6e770a1c`) | `vllm/v1/kv_offload/` | Position-aware `CachePolicy.insert(req_context)`; ZMQ DEALER `_SNDHWM=1000` non-blocking send; promotion hold during reset; NIXL rollback; OBJ `blocks_per_chunk` isolation | `100%` running prefix retention under multi-batch LRU; `0` ZMQ hangs; `0` leaked NIXL handles |
| `NVIDIA/TensorRT-LLM` `#18843` (`ba6c1fc87d`), `#21027` (`e7d3441e77`), `#20915` (`732fd63463`) | `tensorrt_llm/_torch/attention/backends/sparse/` & `pyexecutor/sampler/beam_search.py` | Modular Mamba/GDN/KDA replay cache + `_TMA_MIN_BACKING_BYTES = 128 KiB`; `PendingBeamHistories` async gather before forward pass | Eliminates speculative replay drift & Blackwell sparse TMA faults; provides exact per-beam histories to logits processors |

---

## 2. Implementation Deep Dive

### 2.1 SGLang Router Confirmed-First Mode & Follower Caps (`#46505` & `#40467`)

When `--cache-pending-prefix-consult-below <ratio>` is configured, `sgl-router` prefers confirmed prefix matches whenever `confirmed_ratio >= consult_below` or `pending_depth <= confirmed_depth`. When a cold prefix (`confirmed_ratio == 0.0`) creates a new pending stake, `--cache-pending-prefix-max-followers <cap>` hides the stake after `cap` follower requests attach, bounding worst-case cold-burst queue amplification across $W$ workers to $(1 + \text{cap})$. Meanwhile, `GET /get_hicache_l3_cache_layout` aggregates Mooncake L3 object keys across DP ranks via `merge_l3_layouts(layouts)`.

### 2.2 vLLM RL `ParamMeta` & Five-Fold KV Offload Hardening (`#63256`, `#61050`, `#63708`, `#62929`, `#63628`, `#63655`)

1. **Position-Aware Running Prefix Preservation (`#61050`)**: `CPUOffloadingManager._record_request_cache_access()` passes `req_context` (implementing `get_offload_key_position(key)`) to `CachePolicy.insert()` and `on_request_keys()`. When evicting keys belonging to the currently running request, keys with higher token positions (tail blocks) are evicted before lower-position prefix blocks, guaranteeing **100% continuous prefix preservation**.
2. **Tiered & P2P Control/Data Plane Safety**: `ZmqConnection` sets `zmq.SNDHWM = 1000` and uses non-blocking `send()` (`#63708`); `TieringManager` sets `_promotions_held = True` during `reset_prefix_cache` until all in-flight promotion `TransferJob`s report (`#62929`); `NixlBackend.add_remote_peer_async` rolls back partial descriptors on failure (`#63628`); and `ObjStorageTier` passes `offloading_spec.blocks_per_chunk` to `FileMapper` (`#63655`).

### 2.3 TensorRT-LLM Speculative GDN/KDA Replay & Beam Search Histories (`#18843`, `#20915`, `#21027`)

`FusedRecurrentGatedDeltaRuleCachedReplayUpdate` commits layer history across all local GDN layers before advancing replay state by the accepted draft + bonus token count, while `csa2/metadata.py` pads staging pools to at least `_TMA_MIN_BACKING_BYTES = 131072` bytes. For Beam Search, `DecoderRunner._start_beam_histories` launches `_gather_beam_path` and an async D2H copy (`PendingBeamHistories`) on the last PP rank right before the model forward pass, overlapping D2H transfer with GPU compute.

---

## 3. AI Infra Innovators Comparison (Fireworks / Together / Cerebras / Groq)

| Dimension | SGLang + vLLM + TensorRT-LLM | Fireworks AI | Together AI | Groq LPU | Cerebras WSE-3 |
|---|---|---|---|---|---|
| **Cold-Prefix Burst & Global KV Routing** | Confirmed-First + `max_followers` cap + `/get_hicache_l3_cache_layout` | Sticky session routing + distributed prefix index with burst spreading | Tiered cache-aware routing paired with adaptive speculative batching | Deterministic slot scheduling without shared dynamic KV pools | Single wafer-scale memory pool eliminates cross-worker KV skew |
| **Tiered KV Offload & Fault Isolation** | Position-aware LRU prefix protection, ZMQ HWM, NIXL rollback, OBJ chunk isolation | Multi-tier NVMe/DRAM/HBM promotion with automatic peer failover | High-speed RDMA KV pool with strict epoch generation checks | Pure on-chip SRAM; no host DRAM or remote RDMA KV offloading | Pure on-chip SRAM; weights and KV states stay on wafer |
| **Hybrid SSM/Mamba & Speculative Decode** | TRT-LLM modular GDN/KDA replay cache + SGLang `gfx950` fused HIP KDA decode | Fused tree-speculative verification with state rollback | Block-wise state snapshots for hybrid Mamba-Transformer serving | High single-stream token velocity paired with deterministic multi-token draft | Wafer-wide dataflow natively accelerates parallel draft verification |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Exercise**: Compute the maximum single-worker prefill queue latency for a 64-request cold-prefix burst across 8 workers ($T_{\text{prefill}}=120\text{ ms}$) with unbounded followers ($7,680\text{ ms}$) versus `--cache-pending-prefix-max-followers 7` ($960\text{ ms}$, an **8.0x** reduction).
2. **Falsifiable Question**: Under what P2P cluster fan-out does a static `ZmqConnection._SNDHWM = 1000` drop legitimate burst cache-invalidation messages on healthy peers?
