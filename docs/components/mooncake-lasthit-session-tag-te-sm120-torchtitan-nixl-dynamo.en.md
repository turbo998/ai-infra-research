# Deep Dive into Mooncake Store `LastHitOnly` Batch-Probe Leases & Session Write Tags, TransformerEngine `SM120` Gated FP8 Shared-Memory Guardrails, TorchTitan RL YaRN Softmax Scaling, and LMCache/NIXL/Dynamo Hybrid KV Routing

> **Language / 语言**: [English](mooncake-lasthit-session-tag-te-sm120-torchtitan-nixl-dynamo.en.md) | [中文](mooncake-lasthit-session-tag-te-sm120-torchtitan-nixl-dynamo.md)  
> **Phase**: 71 (`priority: 80`, Track C — Distributed Training, KV Storage & RDMA Collectives)  
> **Evidence Tier**: `implementation` (pinned-commit source audit) + `author-benchmark` (author TorchTitan GPT-OSS-20B RL KL divergence reduction `0.055 -> 0.0006`) + `own-offline` (deterministic stdlib verification `experiments/mooncake-lasthit-session-tag-te-sm120-torchtitan-nixl-dynamo/run.py`)

---

## 1. Architectural Context & Core Engineering Questions

Recent commits across distributed KV storage engines (Mooncake, LMCache, Dynamo), RDMA/DOCA runtimes (NIXL), and distributed training/RL frameworks (TransformerEngine, TorchTitan, Megatron-LM) resolve four foundational systems challenges:

1. **Mooncake Store `BatchProbeKey` Over-Leasing vs. Eviction Races (`kvcache-ai/Mooncake@5d77d3446b9f`, `#4314`) and Replica-Committed Session Attribution (`@5b2069a120b2`, `#4385`)**: When a serving engine probes $N$ ordered prefix candidates (each spanning `candidate_size` layer/rank shard keys), granting read leases on every hit pins all intermediate prefixes and starves LRU eviction, whereas lease-free probing leaves the winning prefix vulnerable to eviction before `BatchGet`. How does `ProbeLeaseMode::LastHitOnly` grant read leases only to the last complete candidate while safely falling back if a shard key is evicted between initial probe and lease acquisition?
2. **TransformerEngine `cast_fp8_gated_kernel` `99 KiB` Opt-In Shared-Memory Overflow on `SM120` (`NVIDIA/TransformerEngine@62b13895c2e6`, `#3600`) and Single-Group CuTeDSL `DenseGEMM` FC2 Dispatch (`@0c4d9610ac3b`, `#3666`)**: Why does the TMA gated FP8 cast kernel—designed for `SM100` (`227 KiB` max opt-in SMEM per block)—fail on `SM120` (`99 KiB` max opt-in SMEM) during BF16 backward (`131,296 B`) or FP32 forward/backward (`164,064 B` / `229,600 B`)?
3. **TorchTitan RL vLLM Rollout Wrapper Missing YaRN `softmax_scale` (`pytorch/torchtitan@8a3f5bcaa4fd`, `#5347`) and Unified `torch.accelerator` Streams (`@35f4a2e16658`, `#5450`)**: Why did replacing TorchTitan attention layers with `VLLMAttention` drop model-specific YaRN `mscale` adjustments on `GPT-OSS-20B` and `DeepSeek-V3`, inflating policy-rollout KL divergence to `0.055` (vs. **`0.0006`** after threading `softmax_scale`)?
4. **LMCache Local-DRAM P2P Lookup (`LMCache/LMCache@c90e6862abb0`, `#6014`), NIXL Active-Release Status Preservation (`ai-dynamo/nixl@5b565cb9a6d4`, `#2175`) & GPUNetIO DOCA Rollback (`@42f1714f4605`, `#2052`), and Dynamo Hybrid Mamba/Attention KVCR Geometry Filtering (`ai-dynamo/dynamo@8bf042187a8f`, `#15535`)**: Why did recurrent Mamba state blocks (`block_size=64`) in a hybrid model cause Dynamo's KV router to trigger `cache_owner_fault` and withdraw the worker's entire `CacheOwner`?

---

## 2. Core Implementation Paths & Source-Level Mechanics

### 2.1 Mooncake Store `LastHitOnly` Reverse Recheck Leasing (`#4314`) & Session Write Tags (`#4385`)

1. **`MasterService::BatchProbeKey` with `ProbeLeaseMode::LastHitOnly` (`kvcache-ai/Mooncake@5d77d3446b9f`, `#4314`)**:
   - **Contract**: Validates `candidate_size > 0` and `keys.size() % candidate_size == 0` (returning `ErrorCode::INVALID_PARAMS` per key otherwise).
   - **Phase 1 (Lease-Free Probe)**: Probes all keys via `ProbeSingleKey(key, /*grant_lease=*/false)` to populate initial existence results without extending TTLs.
   - **Phase 2 (Reverse Candidate Recheck & Lease Grant)**: Scans `candidate_size` groups in reverse (`end = keys.size(); end > 0; end -= candidate_size`). Groups with any `false` entry are skipped immediately. For an all-`true` group, the Master re-acquires each key's shard lock and calls `ProbeSingleKey(key, /*grant_lease=*/true)`, overwriting `results[i]` with fresh recheck status. If a concurrent eviction invalidated a member key, that candidate is no longer all-`1` and the reverse scan seamlessly falls back to the next earlier complete candidate; once a group rechecks all-`true`, the scan terminates (`break`).
2. **Replica-Committed `session_ids` Write Attribution (`@5b2069a120b2`, `#4385`)**:
   - Adds `session_ids` (`<= 256` B UTF-8 strings) to `ReplicateConfig` and `Metadata`. `PutStart` stages a `PendingSessionTag` scoped to `eligible_replica_ids`; `PutEnd` commits it only when an eligible replica completes, while `PutRevoke` clears it via `ClearSessionTagIfNoViableReplica`.

### 2.2 TransformerEngine `SM120` Gated FP8 Shared-Memory Guard (`#3600`) & CuTeDSL FC2 Dispatch (`#3666`)

1. **`cast_fp8_gated_kernel` Dynamic + Static Shared-Memory Fit (`NVIDIA/TransformerEngine@62b13895c2e6`, `#3600`)**:
   - With double buffering (`BUFFERS_NUM = 2`), $64 \times 128$ tiles (`16,384` elements), `TMA_SHMEM_ALIGNMENT = 128 B`, and `96 B` of static shared memory (`32 B` mbarriers + `64 B` `reduce_max` workspace), total per-block shared memory is:
     $$
     S_{\text{total}} = S_{\text{grad}} + 2 S_{\text{in}} + 2 S_{\text{out}} + 128\text{ B} + 96\text{ B}
     $$
   - Comparing against `SM100` (`227 KiB = 232,448 B`) vs. `SM120` (`99 KiB = 101,376 B`):
     - **BF16 Forward (`in=2B, out=1B`)**: $S_{\text{total}} = 98,432 + 96 =$ **`98,528 B`** $\le 101,376\text{ B}$ (fits `SM120` TMA);
     - **BF16 Backward (`grad=2B, in=2B, out=1B`)**: $S_{\text{total}} = 131,200 + 96 =$ **`131,296 B`** $> 101,376\text{ B}$ (fits `SM100`, falls back to non-TMA kernel on `SM120`);
     - **FP32 Forward / Backward (`in=4B`)**: **`164,064 B`** and **`229,600 B`** (exceed `SM120` limit; automatically routed to non-TMA kernels).
2. **Single-Group CuTeDSL `DenseGEMM` FC2 Without `columnwise_data` (`@0c4d9610ac3b`, `#3666`)**:
   - Enables `_single_group_fc2_gemm` in `GroupedMLP` whenever `not weight_requires_grad` (`torch.no_grad()` or frozen expert weights), even when backward `columnwise_data` was not allocated.

### 2.3 TorchTitan RL YaRN `softmax_scale` Preservation (`#5347`) & Megatron-LM DSA Spec Cloning (`#8414`)

1. **TorchTitan RL `Attention.Config.softmax_scale` (`pytorch/torchtitan@8a3f5bcaa4fd`, `#5347`)**:
   - Exposes `softmax_scale` on `Attention.Config`:
     $$
     \text{softmax\_scale} = d_{qk}^{-0.5} \times \left(0.1 \times \text{mscale} \times \ln(\text{rope\_factor}) + 1.0\right)^2
     $$
   - Yields **`0.2266575543`** for `gpt_oss` ($d_{qk}=64, \text{rope\_factor}=32.0$) and **`0.1142519529`** for `deepseek_v3` ($d_{qk}=192, \text{rope\_factor}=40.0, \text{mscale}=0.7$), reducing GPT-OSS-20B RL policy-rollout KL divergence from `0.055` to **`0.0006` (`91.7x` reduction)**.
2. **Megatron-LM Experimental DSA `dataclasses.replace` (`NVIDIA/Megatron-LM@a747bac258df`, `#8414`)**:
   - Clones the cached `ModuleSpec` via `dataclasses.replace` when patching DSA attention so subsequent non-DSA callers are not mutated.

### 2.4 LMCache Local-DRAM P2P (`#6014`), NIXL Active-Release Guard (`#2175`, `#2052`), and Dynamo Hybrid KVCR Filter (`#15535`)

1. **LMCache P2P Lookup Restricted to `(DRAM, LOCAL)` L1 (`LMCache/LMCache@c90e6862abb0`, `#6014`)**: Sets `allowed_l1_memory_tags=frozenset({MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)})` and `allowed_l2_memory_tags=frozenset()` in `p2p_lookup_and_lock` so peer probes only lock chunks resident in the local DRAM L1 registered with the transfer channel.
2. **NIXL `releaseXferReq` Active-Status Preservation (`ai-dynamo/nixl@5b565cb9a6d4`, `#2175`) & GPUNetIO OOB Validation (`@42f1714f4605`, `#2052`)**:
   - Keeps `req->status = NIXL_IN_PROG` when `releaseReqH` rejects releasing an active transfer (`NIXL_ERR_REPOST_ACTIVE`) so subsequent `getXferStatus` calls still enter `checkXfer`; validates OOB IPv4 before GPU allocations and rolls back `doca_dev_close -> doca_verbs_pd_destroy -> doca_verbs_context_destroy` in strict order on failure.
3. **Dynamo KV Router Hybrid Model `NonMainAttentionGroup` Filtering (`ai-dynamo/dynamo@8bf042187a8f`, `#15535`)**:
   - Runs `preprocess_residency_with_reason` *before* the `*block_size != kv_block_size` check in `publisher.rs`, cleanly skipping `NonMainAttentionKind` / `NonMainAttentionGroup` Mamba events instead of faulting the worker's `CacheOwner`.

---

## 3. Deterministic Offline Verification (`own-offline`)

Experiment script: `experiments/mooncake-lasthit-session-tag-te-sm120-torchtitan-nixl-dynamo/run.py` (results sealed in `summary.json` and guarded by `tests/test_phase71_experiments.py`):

| Verification Target | Key Inputs / Parameters | Measured / Simulated Output | Architectural Conclusion |
|---|---|---|---|
| Mooncake `LastHitOnly` & Race Fallback | 4 candidates (`size=2`), `[T,T, F,F, T,T, T,F]`; simulated $p_2$ eviction | Normal selects `cand=2` (rechecks `[2]`); under $p_2$ race falls back to `cand=0` | Eliminates prefix over-leasing and safely degrades to earlier complete prefix on race |
| TE `cast_fp8_gated_kernel` SMEM Fit | `SM100` (`227 KiB`) vs `SM120` (`99 KiB`), `static=96B` | `fwd_bf16=98,528 B` (fits SM120); `bwd_bf16=131,296 B` / `fp32=164,064 B` (fallback) | Including `96 B` static SMEM prevents `SM120` TMA launch failures |
| TorchTitan RL YaRN `softmax_scale` | GPT-OSS ($d=64, f=32$) / DeepSeek-V3 ($d=192, f=40, m=0.7$) | `0.2266575543` / `0.1142519529`; KL drops `0.055 -> 0.0006` (`91.67x`) | Eliminates policy-rollout attention temperature mismatch in RL |
| LMCache / NIXL / Dynamo Guardrails | Hybrid KVCR `(attn=16, mamba=64)`; NIXL rejected active release | Dynamo zero false faults; NIXL keeps `NIXL_IN_PROG`; LMCache locks `(DRAM, LOCAL)` only | Prevents false `CacheOwner` withdrawal on hybrid models and NIXL polling deadlocks |

---

## 4. Solutions Architect Exercises & Falsifiable Questions

1. **Calculation Exercise**: In TransformerEngine's `cast_fp8_gated_kernel`, if input activations are `FP8` (`1 B`), gradients are `BF16` (`2 B`), and outputs are `FP8` (`1 B`) with double-buffered $64 \times 128$ tiles and `96 B` static SMEM, compute the total backward shared-memory requirement and determine whether it fits `SM120` (`99 KiB = 101,376 B`).
   - *Solution*: $S_{\text{grad}} = 32,768\text{ B}$, $2 S_{\text{in}} = 32,768\text{ B}$, $2 S_{\text{out}} = 32,768\text{ B}$, giving $S_{\text{total}} = 98,304 + 128 + 96 = 98,528\text{ B} \le 101,376\text{ B}$. Yes, it fits `SM120`.
2. **Falsifiable Hypothesis**: If a candidate group in Mooncake's `LastHitOnly` reverse scan acquires read leases on its first two keys before discovering its third key was evicted, must the Master revoke those two partial leases before falling back to an earlier candidate?
   - *Falsification*: No—read leases only grant temporary TTL eviction protection without mutating object data. Updating the missing key's result to `0` disqualifies the incomplete candidate and falls back to the earlier complete candidate; the partial leases expire naturally at TTL.
