# SGLang Detokenizer Batch Output Grouping, Radix Tree Path Child Count & Dynamo Pre-Crash CUDA Coredump Wait Gate

> **中文版**: [sglang-detok-batch-outputs-radix-path-child-count-dynamo-coredump.md](sglang-detok-batch-outputs-radix-path-child-count-dynamo-coredump.md)

## 1. Primary Source Evidence & Core Claims

| Repository & Commit | Evidence Mode | Source Paths | Mechanism & Quantitative Finding |
|---|---|---|---|
| `sgl-project/sglang` `#45280` (`468e26f253`, 2026-10-20) | `implementation` | `python/sglang/srt/environ.py`, `python/sglang/srt/managers/multi_tokenizer_mixin.py` | Adds `SGLANG_BATCH_TOKENIZER_OUTPUTS` (default `True`) in `_send_output_to_tokenizer_workers` to group non-contiguous batch outputs by tokenizer worker in owner-arrival order, with opt-out (`False`) for per-request delivery. |
| `sgl-project/sglang` `#45241` (`dc62c3effd`, 2026-10-20) | `implementation` | `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py`, `rust/sglang-radix-tree/src/{unified_tree_core.rs,python_bindings.rs}` | Extends `TreePathNodeInfo` in Python and Rust to a 6-tuple `(node_id, parent_id, path_start, path_end, full_value_len, num_children)` reporting immediate child count at snapshot time. |
| `ai-dynamo/dynamo` `#16500` (`b169bcc08a`, 2026-10-20) | `implementation` | `components/src/dynamo/sglang/args.py`, `components/src/dynamo/sglang/tests/test_sglang_unit.py` | Adds `_default_cuda_coredump_wait(server_args)` to set `SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS="0"` when user-triggered CUDA coredumps and `--crash-dump-folder` are disabled, eliminating the 60s `SIGQUIT` exit delay. |

## 2. Implementation Path, Data Structures & State Machines

### 2.1 Multi-Tokenizer Batch Return Grouping (`#45280`)
When `SGLANG_BATCH_TOKENIZER_OUTPUTS=True`, `_send_output_to_tokenizer_workers` calls `_split_batch_by_key(output, ipcs)` to coalesce non-contiguous request outputs belonging to the same IPC worker into a single ZMQ message (reducing 12 sends across 3 workers to 3 sends, `4.0x` fewer ZMQ IPC calls) while preserving first-seen owner order. Setting `SGLANG_BATCH_TOKENIZER_OUTPUTS=False` dispatches each request individually via `_handle_output_by_index`.

### 2.2 Unified Radix Tree `TreePathNodeInfo` 6-Tuple (`#45241`)
Both Python `UnifiedTreeCore.collect_path_info` and Rust `sglang-radix-tree` append `num_children` (`len(node.children)`) to `TreePathNodeInfo`, transitioning a shared prefix head node from `num_children=1` to `num_children=2` upon branch insertion while keeping prior snapshots immutable.

### 2.3 Dynamo Pre-Crash CUDA Coredump Wait Gate (`#16500`)
`_default_cuda_coredump_wait` sets `SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS="0"` unless the variable is already set, `CUDA_ENABLE_USER_TRIGGERED_COREDUMP=="1"`, or `server_args.crash_dump_folder` is non-empty, saving 60 seconds of dead-worker stall on `SIGQUIT`.

## 3. Deterministic Verification

- **Verifier**: `python experiments\sglang_detok_batch_radix_child_count_dynamo_coredump\run.py --check`
