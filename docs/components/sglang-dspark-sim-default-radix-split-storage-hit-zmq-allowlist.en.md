# SGLang DSpark Simulated Acceptance Opt-In Default, RadixCache `UpdatePendingBackupOnNodeSplit`, Storage-Tier Waiting Prefill Load Deduction, and ZMQ ZAP Peer Allowlist

> **Language / 语言**: [中文 (`sglang-dspark-sim-default-radix-split-storage-hit-zmq-allowlist.md`)](sglang-dspark-sim-default-radix-split-storage-hit-zmq-allowlist.md) | **English (Current)**
> **Evidence Class**: `implementation` (upstream commit verification) + `own-offline` (deterministic stdlib simulator)
> **Upstream Anchors**:
> - `sgl-project/sglang@f0b65907227480afc376a66db62931b7e9c46c38` (`2026-10-20`, PR `#45433`)
> - `sgl-project/sglang@cc7ae372ac2fba9605460308c8f347c80702c828` (`2026-10-20`, PR `#45242`)
> - `sgl-project/sglang@0d7638b03509967e749c201146bfb04bbdfe380c` (`2026-10-20`, PR `#45158`)
> - `sgl-project/sglang@194876bea07214539306526f1bd937a96e02bff9` (`2026-10-20`, PR `#45250`, `CVE-2026-93034`)
> - `sgl-project/sglang@468e26f253b2f1d42bf3545742b7c363d0c6592e` (`2026-10-20`, PR `#45280`)

---

## 1. Architectural Findings & Root-Cause Analysis

### 1.1 DSpark Simulated Acceptance Shortcuts Disabled by Default (`f0b6590`)

- In `python/sglang/srt/environ.py`, `SGLANG_SIMULATE_ACC_GREEDY` and `SGLANG_DSPARK_SIMULATE_FOLD` previously defaulted to `EnvBool(True)`, causing temperature-sampled requests under `SGLANG_SIMULATE_ACC_LEN` to silently take greedy draft/accept shortcuts or graph-folded verify epilogues unless overridden.
- Commit `f0b6590` flips both defaults to `EnvBool(False)`, making simulated acceptance shortcuts strictly opt-in.

---

### 1.2 UnifiedRadixCache `UpdatePendingBackupOnNodeSplit` Component Action (`cc7ae37`)

When a radix tree node splits (`parent -> old_node => parent -> new_node -> new_child`):
- **Pre-fix defect**: `ReplaceWriteThroughOnNodeSplit` was only emitted when `child.write_through_pending_id is not None`. In **host-memory buffer-only mode** (`is_host_memory_buffer_only=True`) or when `SWAComponent._maybe_split_leaf_for_swa_lock` caps a fresh SWA leaf at one window (`ack_id is None`), `FullComponent`'s buffer-write dependency tracker never received the split notification, leaving orphaned dependencies.
- **Fix (`cc7ae37`)**:
  1. Replaces `ReplaceWriteThroughOnNodeSplit` with `UpdatePendingBackupOnNodeSplit(ack_id: int | None, old_node_id, new_node_id, new_child_node_id, component_type=ComponentType.FULL)`.
  2. `UnifiedTreeCore._split_node` emits `UpdatePendingBackupOnNodeSplit` whenever `ack_id is not None or self.is_host_memory_buffer_only`, and `SWAComponent._maybe_split_leaf_for_swa_lock` appends any returned action to `cache_actions`.
  3. `UnifiedRadixCache._apply_cache_action` relocates write-through and linker pending offload lists when `ack_id is not None` and then routes the action to `FullComponent.apply_component_action` (`1 -> 0` orphaned buffer dependencies).

---

### 1.3 Excluding Accepted Storage-Tier Hits from Waiting Prefill Load (`0d7638b`)

In `SchedulerLoadInquirer.get_num_waiting_uncached_tokens()`:
- **Pre-fix defect**: Waiting requests only deducted `req.num_matched_prefix_tokens` (device/host prefix cache hits), counting tokens already accepted for storage-tier prefetch as uncached prefill work and inflating reported prefill queue load.
- **Fix (`0d7638b`)**:
  1. `UnifiedRadixCache.waiting_storage_span(request)` returns the logical token span `(storage_start, storage_start + storage_hit_count)` for accepted `ongoing_prefetch` entries (`info.host_indices is not None`) or `(matched_len, matched_len + num_tokens)` for `buffer_pipeline.staged_prefetches` (returning `None` for unaccepted query results where `host_indices is None`).
  2. `get_num_waiting_uncached_tokens()` subtracts the non-overlapping storage hit span:
     $$\text{storage\_tokens} = \max\!\left(0,\; \min(\text{span}_1, \text{seqlen}) - \max(\text{span}_0, \text{matched})\right), \quad \text{uncached} \mathrel{-}= \text{storage\_tokens}$$
  3. **Quantitative Impact (`own-offline`)**: Across a 3-request mix (`4096`-token accepted ongoing prefetch, `8192`-token staged prefetch, and `2048`-token unaccepted query), reported waiting uncached tokens drop from `12,544` to `3,328` (**-73.47%** false prefill load inflation) while keeping unaccepted queries counted until admission.

---

### 1.4 TCP-Bound ZMQ IPC Socket ZAP Peer Allowlist (`194876b`, `CVE-2026-93034`) and Detokenizer Batch Toggle (`468e26f`)

- **ZMQ ZAP Peer Allowlist (`194876b`)**: Mitigates `CVE-2026-93034` via `SGLANG_ZMQ_ALLOWED_PEERS` and `restrict_zmq_peers(context, socket)` on TCP-bound ZMQ sockets (`data_parallel_controller.py` and `load_snapshot.py`). Shares one `ThreadAuthenticator` per `context.underlying`, normalizes IPv4/IPv6 strings via `ipaddress.ip_address(p).compressed` to match `libzmq` ZAP formatting, and sets `socket.zap_domain = b"sglang"` prior to `bind()`.
- **Detokenizer Grouping Toggle (`468e26f`)**: Adds `SGLANG_BATCH_TOKENIZER_OUTPUTS` (default `True`) to allow opting out of `_split_batch_by_key` grouped IPC sends when per-request streaming latency is prioritized.

---

## 2. Deterministic Verification (`own-offline`)

- **Experiment Script**: [`experiments/sglang_dspark_default_radix_split_storage_load_zmq/run.py`](../../experiments/sglang_dspark_default_radix_split_storage_load_zmq/run.py)
- **Command**: `python experiments/sglang_dspark_default_radix_split_storage_load_zmq/run.py --check`
- **Verified Invariants**:
  - `dspark_simulated_shortcuts_default_off = true`
  - `radix_split_routes_both_wt_and_buffer_only = true` (`0` orphaned buffer dependencies)
  - `storage_tier_waiting_load_deducts_only_accepted = true` (`12,544 -> 3,328` tokens, `-73.47%` false load inflation)
  - `zmq_zap_blocks_unauthorized_peers = true`
