# SGLang PD+DP Two-Phase Role Switch, TRT-LLM Attention-DP KV Event Collective Alignment & Mooncake Store Zero-Copy VMM

Deterministic Python 3.12 standard-library model reproducing:
1. **SGLang PR #44215 (`af49ada16d`)**: Enables runtime Prefill/Decode role switching under multi-rank Data Parallelism (`dp_size > 1`) and DP Attention (`--enable-dp-attention`, rejecting only `--enable-dp-attention-local-control-broadcast`) by introducing a two-phase `prepare_only=True` preflight check across all DP workers before any worker tears down its KV/transfer/CUDA-graph state.
2. **TensorRT-LLM PR #20314 (`ec782566f5`)**: Moves `_add_kv_cache_events()` outside `_process_previous_batch()` in `_executor_loop_overlap` and runs `_consume_previous_batch_for_rebalance()` across all Attention-DP (ADP) ranks even when `previous_batch is None`, preventing `kv_event_allgather` collective deadlocks across `("absent", "processed")`, `("processed", "deferred")`, and `("deferred", "absent")` rank states.
3. **TensorRT-LLM PR #20108 (`69e06b8afb`)**: Enables zero-copy GPUDirect RDMA (`stage_through_host=False` default) in the Mooncake Store KV connector by merging overlapping KV regions (`merge_intervals`) and cutting virtual VMM reservations at `cuMemCreate` physical allocation boundaries (`allocation_spans` / `split_at_cuts`), eliminating `1024 MiB` of pinned host staging DRAM on an 8-GPU node.

## Run

```bash
python experiments/sglang_pd_dp_role_switch_trtllm_adp_kv_mooncake_zc/run.py --check
```
