# Disaggregated Prefill/Decode KV Routing & Hierarchical Cache Model (`own-offline`)

This CPU-only, standard-library experiment (`experiments/disaggregated_kv_routing/run.py`) models three primary-source mechanisms across Mooncake, NVIDIA NIXL, and SGLang:

1. **KV Layout, NIXL Strided Descriptor Compression (`nixlStrideDesc`), and Mooncake RDMA Slice Tail-Folding (`planRdmaSlices`)**:
   - Compares per-rank KV transfer payloads and scatter-gather descriptor counts across `Llama-3-70B-GQA-BF16` (`TP=8`), `Qwen3-32B-GQA-FP8` (`TP=4`), and `DeepSeek-V3-MLA-FP8` (`TP=1`).
   - Exact port of `mooncake::tent::planRdmaSlices` (`mooncake-transfer-engine/tent/include/tent/transport/rdma/slice.h:69-90`), demonstrating `merge_ratio = 0.25` tail-folding (`16 -> 15` slices when the 16th tail slice is `< 25%` of `block_size`).
2. **Mooncake Deadline-Aware MLU NIC Arbitration (`DeadlineMlu` & `OrderByUrgency`)**:
   - Exact port of `mooncake::tent::DeadlineMlu` (`deadline_mlu.h:56-66`) and `mooncake::tent::OrderByUrgency` (`bw_arbitration.h:77-116`), comparing FIFO, equal-bandwidth sharing, and greedy cumulative `bytes_ahead` urgency ordering under 4-flow and 6-flow `388 Gbps` NIC contention.
3. **SGLang 3-Tier Decode HiCache Overlap (`DecodePrefixMatch` + `HiCacheRestoreGatedKVReceiver`)**:
   - Models `decode_prefix_len = l1 + l2 + l3`, `restore_token_count = decode_prefix_len - l1`, and `max(remote_prefill_rdma_branch, local_l3_l2_restore_branch)` from `python/sglang/srt/disaggregation/decode_hicache_mixin.py:26-51, 166-181`.

## Commands

```bash
python experiments/disaggregated_kv_routing/run.py
python experiments/disaggregated_kv_routing/run.py --check
```
