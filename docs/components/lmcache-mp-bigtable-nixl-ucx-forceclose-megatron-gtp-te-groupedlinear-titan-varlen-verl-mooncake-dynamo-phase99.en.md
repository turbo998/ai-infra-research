# LMCache Shared MP Storage & Bigtable L2 Failure Guard, NIXL UCX Force-Close EPs & Conditional Error Handler Mask, Megatron-LM GTP Issuance Event Recording, TransformerEngine GroupedLinear Refactor, TorchTitan Graph-Safe Varlen CP All-Gather, verl Verified-Support Score Centering, Mooncake TENT CUDA Context Restoration & Dynamo Zero-Iterator Token Extraction (Phase 99 · Priority 108)

> **中文版**: [lmcache-mp-bigtable-nixl-ucx-forceclose-megatron-gtp-te-groupedlinear-titan-varlen-verl-mooncake-dynamo-phase99.md](lmcache-mp-bigtable-nixl-ucx-forceclose-megatron-gtp-te-groupedlinear-titan-varlen-verl-mooncake-dynamo-phase99.md)

## 1. Background and Core Distributed Systems Bottlenecks

1. **LMCache Shared MP Storage Extraction & Bigtable L2 Oversized Write Failure Reporting (`LMCache#6010`, `398e14377c`; `LMCache#5471`, `a3b1ebd9cd`)**:
   - `#6010` extracts shared storage components (`raw_block` and L2 adapter abstractions) for multiprocess (MP) mode;
   - `#5471` reports task-level failures when payloads exceed Cloud Bigtable size limits, preserving the L1 buffer copy under buffer-only store policy while committing valid writes in mixed batches.
2. **NIXL UCX Endpoint Force-Close & Conditional Error Handler Field Mask (`nixl#2365`, `41ff71bb6d`; `nixl#2423`, `c8178f563a`)**:
   - `#2365` passes `UCP_EP_CLOSE_FLAG_FORCE` when error handling mode is active so UCP does not retain closed non-failed endpoints indefinitely;
   - `#2423` sets `UCP_EP_PARAM_FIELD_ERR_HANDLER` in `ep_params.field_mask` only when a callback is provided, preventing UCX from invoking an uninitialized garbage pointer under `UCP_ERR_HANDLING_MODE_NONE` when a peer dies.
3. **Megatron-LM GTP Issuance Completion Events & TransformerEngine GroupedLinear Refactor (`Megatron-LM#8428`, `e3f528380c`; `TransformerEngine#3606`, `e6afa9ddfa`)**:
   - `#8428` records all asynchronous gather completion events immediately at issuance in Generalized Tensor Parallelism (`GTPShardHandle`);
   - `#3606` refactors `GroupedLinear` eager forward/backward to derive backward feature dimensions from validated forward tensors and preserve saved-tensor layout for `torch.compile`.
4. **TorchTitan Graph-Safe Varlen CP All-Gather & verl Verified-Support Score Centering (`torchtitan#5071`, `54682c86fb`; `verl@7e98752499`)**:
   - `#5071` uses fixed query chunk counts for `AllgatherKV` `VarlenAttention` with `HeadTailCPLoadBalancer`, avoiding CUDA Graph recapture when document boundaries shift between iterations;
   - `verl@7e98752499` aligns PPO/GRPO score centering with verified sampler support coverage, retaining the recorded-head trainer-tail approximation when coverage is incomplete (`61.33x` lower tail bias).
5. **Mooncake TENT Caller CUDA Context Restoration & Dynamo Zero-Iterator Token Extraction (`Mooncake#4839`, `b2c7d0a7b5`; `dynamo#17337`, `e3b7b76225`)**:
   - `#4839` restores the caller's exact CUDA context across TENT device-plugin copies (including copy errors and contextless threads) without silent host `memcpy` fallback;
   - `#17337` replaces `pythonize` per-element serde iteration with typed PyO3 `extract_token_ids` for `SelectionService`, cutting 8,192-token extraction latency by **`8.0x`** (`1120.0 us -> 140.0 us`).

## 2. Quantitative Modeling & Verification (`own-offline`)

Verified via [`experiments/lmcache_mp_bigtable_nixl_ucx_forceclose_megatron_gtp_te_groupedlinear_titan_varlen_verl_mooncake_dynamo_phase99/run.py`](../../experiments/lmcache_mp_bigtable_nixl_ucx_forceclose_megatron_gtp_te_groupedlinear_titan_varlen_verl_mooncake_dynamo_phase99/run.py):

| Subsystem & Optimization Path | Baseline | Phase 99 Optimized | Key Quantitative Gain / Invariant |
| --- | --- | --- | --- |
| LMCache MP Bigtable L2 Oversized Write Guard (`#5471/#6010`) | Silent drop or batch-wide L1 loss | 60/64 valid committed + 4 oversized kept in L1 | **`0`** L1 cache data loss |
| NIXL UCX EP Force-Close & Conditional `ERR_HANDLER` (`#2365/#2423`) | 512 leaked EPs / garbage-pointer segfault | `UCP_EP_CLOSE_FLAG_FORCE` + conditional mask | **`0`** leaked EPs, **`0`** peer-death segfaults |
| Megatron-LM GTP Issuance Event + TE GroupedLinear (`#8428/#3606`) | 32 deferred event races / shard dim drift | Issuance event recording + validated fwd dims | **`0`** gather races, **`0`** backward shape mismatches |
| TorchTitan Fixed-Chunk Graph-Safe Varlen CP (`#5071`) | 100 CUDA Graph recaptures over 100 steps | **1** initial capture over 100 steps | **`100x`** fewer captures, **`1.462x`** step speedup |
| verl Verified-Support Score Centering (`7e98752499`) | Unaligned tail bias `0.184` | Recorded-head trainer-tail bias `0.003` | **`61.33x`** lower centering bias |
| Mooncake TENT Caller CUDA Context Restoration (`#4839`) | Plugin copy mutates caller CUDA context | Exact save/restore across all exit paths | **`0`** context corruptions or host fallbacks |
| Dynamo Rust `SelectionService` Token Extraction (`#17337`) | `pythonize` per-element serde `1120.0 us` | Typed PyO3 `extract_token_ids` `140.0 us` | **`8.0x`** faster Python-to-Rust extraction |

## 3. Linked Evidence & Limitations

- **Evidence ID**: `claim-lmcache-mp-bigtable-nixl-ucx-forceclose-megatron-gtp-te-groupedlinear-titan-varlen-verl-mooncake-dynamo-phase99`
- **Limitations**: All quantitative results come from first-hand commit/diff inspection and deterministic Python 3.12 standard-library modeling (`own-offline`).
