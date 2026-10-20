# vLLM V1 UMBP Shared KV Connector (`UMBPStoreConnector`) Topology/Layout Contracts and THUDM/slime Straw In-Place Checkpoint Rewind (`rewind.py`)

> **中文版**: [vllm-v1-umbp-kv-connector-slime-straw-rewind.md](vllm-v1-umbp-kv-connector-slime-straw-rewind.md) | **Associated Experiment**: [`experiments/vllm_umbp_kv_connector_slime_rewind/README.md`](../../experiments/vllm_umbp_kv_connector_slime_rewind/README.md)

### One-Sentence Takeaway

vLLM V1 established a **client-agnostic Unified Memory Buffer Pool (UMBP) shared KV connector contract** in `vllm/distributed/kv_transfer/kv_connector/v1/umbp/` (`#61790`): it combines a `[PP, PCP, TP]` `RankTopology` where `DCP` partitions `TP` (`dcp_rank = tp_rank % dcp_size`, avoiding a 4th Cartesian worker dimension), a worker-derived 32-hex-char SHA-256 `UMBPNamespace` handshake (`layout_version: 2`, per-layer `KVCacheSpec` fingerprints, and EAGLE draft-model isolation), a multi-part `KVLayoutPlanner` for oversized layer pages, a scheduler (`UMBPStoreConnectorScheduler`) that preserves at least the final prompt token (`max_external_token = num_tokens - 1`), drops one block for EAGLE, requires all-rank hits per block, and pins exported GPU blocks until all workers complete (`_StoreEvent`), and a worker (`UMBPStoreConnectorWorker`) that enforces atomic batch publication (`TransferJobStatus.COMPLETED`) and synchronous preemption barriers (`handle_preemptions`); in parallel, THUDM/slime (`#2482`) implemented **in-place Straw checkpoint rewind** in `slime/data/{rewind.py,checkpoint.py}`, writing an immutable `rollout/committed_{step}.json` after reverse-depth directory `fsync`, clamping crash recovery to $\min(\text{tracker}, \max(\text{commits}))$, regenerating in-flight prompt groups via a four-tier fallback (including exact cross-epoch dataset cursor arithmetic $\text{divmod}(\text{epoch}\cdot |\mathcal{D}| + \text{offset} - \Delta, |\mathcal{D}|)$), releasing `step_id > step` storage owners (`discard_future`), and atomically replacing the queue WAL (`reset_journal`).

### Component Responsibilities and Upstream Locations

| Module & File | Upstream Repository & Commit / Blob SHA | Core Engineering Responsibility |
| :--- | :--- | :--- |
| `vllm/distributed/kv_transfer/kv_connector/v1/umbp/data.py` | `vllm-project/vllm` commit `82b7dad5e315040219ec657398d6197010121d15` / blob `9374c9ced8020e85b006e0ebd8ea154bddbfc658` | Defines `RankTopology`, `UMBPNamespace`, `BlockIdentityCodec`, `KVRegion`/`KVLayoutPlanner`, `RequestTracker`, `TransferJobState`, and `UMBPConnectorWorkerMetadata`. |
| `vllm/distributed/kv_transfer/kv_connector/v1/umbp/{scheduler.py,worker.py,connector.py}` | `vllm-project/vllm` commit `82b7dad5e315040219ec657398d6197010121d15` / blobs `dd7ecadb16`, `34101f294c`, `e828ac4ee0` | Implements scheduler all-rank prefix matching, EAGLE tail-block drop, async-load forwarding, and `_StoreEvent` refcounted block freeing, plus worker key localization, atomic publish, and preemption barriers. |
| `slime/data/rewind.py` | `THUDM/slime` commit `4f204660abcd531882dd4918ee301f107a952eb4` / blob `e1e053461f8e18672797b93c4602fb4a478f62c2` | Implements `restore_dataset_cursor`, `fresh_samples`, the four-tier `regeneration_groups` prompt recovery hierarchy, `reset_journal` atomic WAL replacement, and `discard_future`. |
| `slime/data/checkpoint.py` | `THUDM/slime` commit `4f204660abcd531882dd4918ee301f107a952eb4` / blob `03b1eede3e7c61cce7a6841bda812fdc77e4fde0` | Implements `save_checkpoint`, `commit_checkpoint` (file + reverse-depth directory `fsync` and SHA-256 digests), `_select_checkpoint`, and `resolve_checkpoint`. |

### Core Implementation Paths, Data Layout, and Synchronization Contracts

1. **vLLM V1 UMBP Rank Topology, Worker Handshake Namespace, and Multi-Part Layout Planner (`data.py:52-415`)**:
   - **`DCP` Partitioning Inside `RankTopology`**: Engine workers are laid out as `[PP, PCP, TP]` with `TP` innermost (`tp_rank = rank % tp_size`, `pcp_rank = (rank // tp_size) % pcp_size`, `pp_rank = (rank // (tp_size * pcp_size)) % pp_size`). `DCP` (`decode_context_parallel_size`) partitions the TP workers (`dcp_rank = tp_rank % dcp_size`, requiring `tp_size % dcp_size == 0` when `dcp_size > 1`) instead of adding a 4th Cartesian dimension. Thus `all_namespaces()` yields exactly $\text{PP} \times \text{PCP} \times \text{TP}$ tuples, formatted as `umbp:vllm:v1:{namespace}:tp{tp}:pcp{pcp}:dcp{dcp}:pp{pp}:g{group_id}:{block_hash.hex()}`.
   - **Worker-Derived `UMBPNamespace` Handshake**: Because the scheduler's `KVCacheConfig` flattens `UniformTypeKVCacheSpecs` groups, workers derive the 32-hex-char SHA-256 namespace from per-layer `_spec_fingerprint`s, `(tp, pp, pcp, dcp)`, `layout_version: 2`, and optional `speculative` draft-model config, reporting it via `UMBPHandshakeMetadata`. `set_xfer_handshake_metadata` verifies that all workers agree on a single namespace before enabling scheduler lookups.
   - **Multi-Part `KVLayoutPlanner`**: When a layer's `region.block_bytes > physical_stride`, `_ranges_for_block` splits each logical `block_id` into `physical_parts = region.block_bytes // physical_stride` contiguous `KVRange` slices (`physical_block_id = block_id * physical_parts + part`).
2. **Scheduler Final-Token Guard, `_StoreEvent` Refcounting, and Worker Atomic Publish / Preemption Barrier (`scheduler.py:118-395`, `worker.py:110-220`)**:
   - **Final-Token Execution Guard & All-Rank Hit Rule**: `get_num_new_matched_tokens` sets `max_external_token = request.num_tokens - 1` so vLLM always executes at least the last prompt token; when `use_eagle` is active, `matched_blocks = max(0, matched_blocks - 1)` so the drafter recomputes the final block. Every block requires all `per_rank = topology.rank_count` rank keys to hit (`all(group_hits[index : index + per_rank])`).
   - **`_StoreEvent` Block Pinning**: `build_connector_meta` calls `gpu_block_pool.touch(...)` on blocks scheduled for store and frees them in reverse order (`reversed(block_ids)`) only when `update_connector_output` observes `event.result.completed_workers >= self._num_workers`.
   - **Atomic Batch Publish & Preemption Drain**: `UMBPStoreConnectorWorker._finish_job` calls `self.runtime.publish(result)` only when `TransferJobStatus.COMPLETED` (withholding the entire batch if any key fails and recording `failed_block_ids`). `handle_preemptions` synchronously calls `self.runtime.wait(job)` on all in-flight loads and stores for preempted requests before vLLM reuses their GPU blocks.
3. **THUDM/slime Straw Joint Commit Boundary and Four-Tier In-Place Rewind (`checkpoint.py:42-265`, `rewind.py:14-145`)**:
   - **Synchronous Joint Commit & Reverse-Depth `fsync`**: `save_checkpoint` forces synchronous actor/critic saves (`force_sync=True`) and `rollout_manager.save`, then `commit_checkpoint` `fsync`s every payload file and parent directory (sorted from deepest to shallowest), records file sizes and SHA-256 digests in `rollout/committed_{id}.json`, and only then invokes `rollout_manager.checkpoint_committed.remote(rollout_id)`.
   - **Four-Tier Prompt Regeneration (`regeneration_groups`)**: On rewind, in-flight queue groups recover clean prompts via: (1) `_queue_original_input` digest-deduped references, (2) untouched `PENDING` samples (`response_length == 0`), (3) `RolloutDataSource` exact dataset cursor rewind ($\Delta = G_{\text{cursor}} - G_{\text{group}}$, $A = E_{\text{cursor}}\cdot |\mathcal{D}| + O_{\text{cursor}} - \Delta$, `divmod(A, |D|)`), or (4) `fresh_samples(group)` stripping generated rollout/training fields.

### Key Formulas, Units, and Engineering Assumptions

1. **UMBP `RankTopology` Mapping and Namespace Cardinality**:
   $$\text{tp\_rank} = r \bmod T, \quad \text{pcp\_rank} = \left\lfloor \frac{r}{T} \right\rfloor \bmod P_{\text{pcp}}, \quad \text{pp\_rank} = \left\lfloor \frac{r}{T \cdot P_{\text{pcp}}} \right\rfloor \bmod P_{\text{pp}}, \quad \text{dcp\_rank} = \text{tp\_rank} \bmod D_{\text{dcp}}$$
   With $T \bmod D_{\text{dcp}} = 0$ for $D_{\text{dcp}} > 1$, total namespaces equal $|\mathcal{N}| = P_{\text{pp}} \times P_{\text{pcp}} \times T$.
2. **Straw Cross-Epoch Dataset Cursor Rewind Formula (`rewind.py:56-68`)**:
   $$\Delta = G_{\text{cursor}} - G_{\text{group}}, \quad A = E_{\text{cursor}} \cdot |\mathcal{D}| + O_{\text{cursor}} - \Delta$$
   $$(E_{\text{recovered}}, O_{\text{recovered}}) = \left(\left\lfloor \frac{A}{|\mathcal{D}|} \right\rfloor, \; A \bmod |\mathcal{D}|\right), \quad I_{\text{recovered}} = I_{\text{cursor}} - \Delta \cdot K_{\text{samples\_per\_prompt}}$$

### Synthetic Experiment and Verification Results (`experiments/vllm_umbp_kv_connector_slime_rewind`)

Running `python experiments/vllm_umbp_kv_connector_slime_rewind/run.py --check` verifies:

| Subsystem & Check | Test Configuration / Input | Deterministic Output / Verified Result | Architectural Significance |
| :--- | :--- | :--- | :--- |
| UMBP `RankTopology` (`rank=5`) | `TP=4, PP=1, PCP=2, DCP=2` | `local_namespace = [1, 1, 1, 0]`, `total_namespaces = 8` | `DCP` partitions `TP` without 16-way Cartesian blowup |
| UMBP Prefix Match Cap (`256` tokens) | `block_size = 64`, `local_tokens = 0` | Standard cap `192` tokens (`3` blocks); EAGLE cap `128` tokens (`2` blocks) | Enforces `max_external_token = 255` so final block executes |
| `KVLayoutPlanner` Multi-Part Split | Layer 0 `64 KiB` + Layer 1 `128 KiB` (`stride=64 KiB`) | Emits `3` `KVRange`s; Layer 1 offsets `[65536, 131072]` | Packs heterogeneous hybrid-attention page sizes per block |
| `_StoreEvent` Refcounted Block Free | `8` workers exporting blocks `[10, 12, 14]` | Freed in reverse order `[14, 12, 10]` after `completed_workers == 8` | Prevents premature block reuse while slow workers export |
| Slime Cross-Epoch Cursor Rewind | $|\mathcal{D}|=100, E=2, O=15, G_{\text{cur}}=215 \to G=195$ | `recovered_epoch = 1`, `offset = 95`, `sample_index = 780` | Exact cross-epoch prompt replay without hook mutation |
| Slime `_select_checkpoint` + `discard_future` | `tracker = 12`, `committed = {8, 10}`, disk has `{8,10,11,12}` | Clamps to `step = 10`, unlinks `{11, 12}` & releases `storage_owner`s | Avoids loading half-saved rollout state after mid-save crash |

### Architect Design Exercises and Falsifiable Questions

1. **Calculation Exercise**: For a 256-token prompt (`block_size = 64`) where 192 tokens are already cached locally on GPU (`num_computed_tokens = 192`), how many external keys does `UMBPStoreConnectorScheduler` query via `runtime.lookup`?
   - *Answer*: `0`. Because `max_external_token = 255` and `range(192 + 64, 256, 64) = range(256, 256, 64)` is empty, no lookup RPC is issued and the final 64 tokens run local prefill.
2. **Falsifiable Hypothesis**: Can an asynchronous RL training framework rely solely on Megatron's `latest_checkpointed_iteration.txt` to resume both model weights and rollout queues after a crash?
   - *Verification*: False (`checkpoint.py:139-174`). Because `actor_model.save_model` updates `latest_checkpointed_iteration.txt` before `rollout_manager.save` and `commit_checkpoint` finish, a crash between model save and rollout commit leaves `tracker > max(commits)`. Recovery must clamp to `min(tracker, max(commits))`.
