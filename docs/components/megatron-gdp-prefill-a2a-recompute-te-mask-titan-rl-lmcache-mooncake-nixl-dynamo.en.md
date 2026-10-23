# Megatron-LM Fused GDP Prefill Prepare & Segmented EP A2A Full Recompute, TE Triton Mask Capacity Drop & Compiled FP8 Weight Cache, TorchTitan Muse Spark GB300 RL 128-GPU Layout, and LMCache/Mooncake/NIXL/Dynamo Data-Plane Evolution

> **Phase 56 (`priority: 65`) · Primary Source & Commit-Level Deep Dive**
> - `NVIDIA/Megatron-LM@fe0e608ca417488cfdeac47b471574c45163360d` (`#6477`)
> - `NVIDIA/Megatron-LM@da2c0ec614d9bb7bdfc16c833ee6568f64f73af9` (`#6429`)
> - `NVIDIA/TransformerEngine@c9c92255465b4518380561b1bbb0aebb177c0b6d` (`#3347`)
> - `NVIDIA/TransformerEngine@9ea05cf2e8dc275d2fd3c88d39062ba20947b77d` (`#3505`)
> - `pytorch/torchtitan@43d1bb98014e0dab19a83e5220ec7ea1ab06d49e` (`#5581`)
> - `pytorch/torchtitan@094c884685c2583323b76be244b76526b85fb520` (`#5152`)
> - `LMCache/LMCache@ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be` (`#5960`), `a81b4e5aea` (`#5922`), `a0ff470128` (`#5976`)
> - `kvcache-ai/Mooncake@dd1217b6c461491e7c73ff29b4ffad215e0a9ac0` (`#4793`), `dae22d472e` (`#4434`), `b66bb2b5f6` (`#4873`)
> - `ai-dynamo/nixl@f3673f6be3f4f80f6e1e38ff86cebcf9af5750d4` (`#2487`), `70badd6aad` (`#2478`)
> - `ai-dynamo/dynamo@3acf24807325febd0462bdff6056e74f17659e71` (`#17143`), `30e972003a` (`#17149`), `c178cbae9b` (`#16501`), `a246c3f227` (`#16351`)

---

## 1. Megatron-LM Fused GDP Prefill Prepare & Segmented EP A2A Overlap Recompute (`#6477`, `#6429`)

- **Gated Delta Product (`GDP`) Fused `gdp_prefill_prepare` (`#6477`)**: Replaces ~**14 separate PyTorch prefill ops** (including 3 strided copies splitting `value/key/query`, folding Householder copies into the sequence dimension, GQA `repeat_interleave`, and computing `beta = sigmoid(b)` and `g = -exp(A_log) * softplus(a + dt_bias)`) with **1 Triton kernel launch**, using `triton.language.extra.libdevice` (`log1p`, `exp`, `div_rn`) for bitwise agreement (`torch.equal`) with the eager reference.
- **Segmented Full Activation Recompute with EP A2A Overlap (`RecomputeSegment`, `#6429`)**:
  - Enables `--recompute-granularity full` (`uniform` and `block`) under `--overlap-moe-expert-parallel-comm` 1F1B schedules.
  - `RecomputeSegment` snapshots the **5 in-place mutated `ModelChunkState` fields** `_MUTABLE_CHUNK_STATE_FIELDS = ("input_ids", "position_ids", "padding_mask", "mtp_input_mask", "mtp_hidden_states")` before the initial `no_grad` forward so MTP `_get_embeddings` does not double-roll `mtp_input_mask` during backward replay.
  - Explicitly detaches `ScheduleNode._forward` outputs when `forward_no_grad = True` so inner `torch.enable_grad()` regions (`bias_dropout_add`) do not pin the forward autograd graph across the forward-to-backward gap.

---

## 2. TransformerEngine Capacity Drop & Compiled FP8 Cache & TorchTitan GB300 RL 128-GPU Layout (`#3347`, `#3505`, `#5581`, `#5152`)

- **TE Triton Mask-Map Capacity Drop (`#3347`) & Compiled FP8 Weight Caching (`#3505`)**:
  - `_row_id_map_pass_2_kernel` clamps destination rows via `row_id = tl.where(row_id >= num_out_tokens, -1, row_id)`, dropping over-capacity tokens instead of writing past the end of `out[num_out_tokens]`.
  - `te.Linear` under `torch.compile` withholds the existing cached workspace when `is_first_microbatch=True` (`not (use_compiled_op and is_first_microbatch)`), creating a fresh FP8 weight cache on the first microbatch and reusing it on subsequent microbatches without mutating functional op inputs.
- **TorchTitan Muse Spark 816B (A42B) GB300 DAPO-Math RL Layout (`#5581`) & Varlen FlashAttention (`#5152`)**:
  - Reconfigures the RL trainer from **256 GPUs (`FSDP=64 × TP=4 × EP=4`)** to **128 GPUs (`FSDP=8 × TP=4 × CP=4 × EP=16`)**: `ContextParallelTransform` maps all layers (including sliding-window layers) to `UlyssesVarlenInnerAttention` (`CP=4`), while `EP=16` (4 GB300 hosts sharing one NVLink domain, `scratch_capacity_factor=16.0`) shrinks the cross-rack expert FSDP all-gather group **8x (from 64 to 8 ranks)**, achieving `186 GiB` peak memory and `15% -> 28%` MFU on a 131K-context GB300 smoke run.

---

## 3. LMCache, Mooncake, NIXL & Dynamo Distributed KV Data-Plane Evolution

- **LMCache Callback `LookupState` (`#5960`), L1/L2 `MemoryTags` Filter (`#5922`) & SWA Prefix Replay Alignment (`#5976`)**: `LookupState` chains `QUERY_PREFETCH_STATUS` immediately upon each server's `LOOKUP` ack via `MessagingFuture.add_done_callback` and frees excess tail locks `[min, hits)` once all servers resolve; under SWA bounded replay (`_prefix_replay_tokens > 0`), `#5976` aligns a full-prompt hit down to `aligned = (ret - 1) // chunk * chunk` and frees `(aligned, ret)` so retrieve never exceeds vLLM's block-floored allocation.
- **Mooncake Single-Clock RDMA Post Chain (`#4793`), Notify Slot Guard (`#4434`) & NVLink IPC Diagnostics (`#4873`)**: Reading `getCurrentTimeInNano()` once per QP WR chain cuts a 256-WR call from **`6.3 us` to `1.9 us`** (`25.0 -> 7.5 ns/WR`, **3.32x speedup**); `sendNotification` advances `notify_send_wr_id_` only after `ibv_post_send` succeeds so a failed post at `wr_id=255` cannot wrap to `256` and overwrite in-flight slot `0`.
- **NIXL UCX 16B `treq` Receiver Correlation (`#2487`) & Repost Telemetry (`#2478`)**: Appends a 16B little-endian `treq` tag (`requestId`, `execution`) after the 26B `tctx` tag (42B total) so `nixl::notif.received` joins with sender phases on `(xfer_request_id, xfer_execution)`.
- **Dynamo 33B Lower-Tier Edge Slots (`#17143`), Scalar Walk (`#17149`), Tiered Session Residency (`#16501`) & SGLang Stream Reader (`#16351`)**: `OwnerTable` interns 24B `IndexedResidencyOwner` into 4B `OwnerId(u32)` via lock-free `ArcSwap<OwnerSnapshot>`, shrinking `EdgeOwnersEntry` from **113 B to 33 B** (**3.42x reduction**).

---

## 4. Deterministic Verification

Run `python experiments/megatron-gdp-prefill-a2a-recompute-te-mask-titan-rl-lmcache-mooncake-nixl-dynamo/run.py --check` to verify all invariants (`summary.json`).
