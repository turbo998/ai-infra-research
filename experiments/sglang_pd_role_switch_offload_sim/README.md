# SGLang Runtime PD Role Switch, PPConsensusStore & Decode KV Offload Simulator (`sglang_pd_role_switch_offload_sim`)

Deterministic Python 3.12 standard-library model of SGLang's runtime Prefill $\leftrightarrow$ Decode role switching (`python/sglang/srt/disaggregation/role_switch.py`), `PPConsensusStore` (`pp_consensus_store.py`), and `DecodeKVCacheOffloadManager` (`decode_kvcache_offload_manager.py`, commit `df13d5dce31abfeff0e701a62e8c39a136758e6c`).

## What It Models

1. **Zero-KV-Reallocation Runtime Role Switching (`role_switch.py`)**:
   - Models the 7-entry table-driven `_reject_reason` precondition check, idle drain gate (`is_fully_idle()`), CUDA Graph headroom verification (`available_graph_gb >= decode_cuda_graph_memory_gb`), and non-rollback unhealthy latching (`_pd_role_switch_unhealthy = True`) on teardown/rebuild exceptions.
2. **`PPConsensusStore` Safe MsgPack Wire Protocol (`pp_consensus_store.py`)**:
   - Models `msgspec.Struct(array_like=True, frozen=True)` 4-element tuple encoding `[kind, rank, key, value]` over ZMQ `PUSH`/`PULL` vs Python `pickle`.
3. **Decode-Side Incremental Page-Aligned KV Offload (`decode_kvcache_offload_manager.py`)**:
   - Models how decode seamlessly recovers the unaligned prompt tail (`len(origin_input_ids) - prefill_offloaded_len`) and offloads incremental chunks at `offload_stride` boundaries while filtering `UnifiedSWAKVPool` sink pages (`swa_indices > 0`).

## Commands

```bash
python experiments/sglang_pd_role_switch_offload_sim/run.py
python experiments/sglang_pd_role_switch_offload_sim/run.py --check
```
