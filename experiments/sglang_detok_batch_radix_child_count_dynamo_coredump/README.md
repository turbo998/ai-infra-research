# SGLang Detokenizer Batch Output Grouping, Radix Tree Path Child Count & Dynamo Pre-Crash Coredump Wait Gate

Deterministic Python 3.12 standard-library model verifying:
1. SGLang PR #45280 (`468e26f253`): `SGLANG_BATCH_TOKENIZER_OUTPUTS` grouped vs per-request ZMQ IPC return dispatch and owner-arrival order preservation.
2. SGLang PR #45241 (`dc62c3effd`): `TreePathNodeInfo` 6-tuple `(node_id, parent_id, path_start, path_end, full_value_len, num_children)` across Python and Rust `UnifiedTreeCore.collect_path_info`.
3. NVIDIA Dynamo PR #16500 (`b169bcc08a`): `_default_cuda_coredump_wait` eliminating the 60-second `SIGQUIT` sleep when user-triggered CUDA coredumps are disabled.
