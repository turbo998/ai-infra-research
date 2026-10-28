# Megatron-LM Single-DP-Group Refit Load Balancing, NCCL 2.32.3-1 Peak BusBW & CFT Counted, TransformerEngine 128x128 MXFP8 Requantize, FlashAttention-4 SM100 Sparse-MLA 64-Head Tail Bound, Mooncake, LMCache, Dynamo & NIXL (Phase 122)

Reproducible stdlib model covering:
- `NVIDIA/Megatron-LM#8502` (`4a55e6fe`): Single-DP-group `TP=1, PP=1` refit source spreading via `src_meta_list[dst_rank % len(src_meta_list)]`.
- `NVIDIA/nccl@53a84c7e` & `NVIDIA/nccl@3a87143f`: NCCL 2.32.3-1 Peak BusBW across GB300 (`NVL4..NVL64`, up to `992 GB/s` Symmetric AllReduce on `NVL64`), B200 (`NVL8`), and H100 (`NVL8`), plus `NCCL_WIN_CFT_COUNTED` counted window operations.
- `NVIDIA/TransformerEngine#3656` (`15dcc848`) & `Dao-AILab/flash-attention@2865a891`: `128x128` persistent grouped MXFP8 requantization (`512 B` contiguous scale blocks, stopping at `live_rows`) and SM100 Sparse-MLA 64-head `lse_log2` tail bound (`seqlen_q` vs `seqlen_q_rounded`).
- `kvcache-ai/Mooncake#4907` (`179ffb21`), `#4903` (`e70c0998`), & `#5008` (`5ef59b0c`): Configurable `max_blocks_per_context`, HCCS `2 MiB` alignment validation, and atomic `activeRanks` view updates.
- `LMCache/LMCache#5858` (`d181fafa`), `ai-dynamo/dynamo#17440` (`b7bf2cc4`), & `ai-dynamo/nixl#2387` (`250581c1`): `aio` gRPC `_STREAM_READ_AHEAD = 1` semaphore (`128 MB -> 4 MB` slow-consumer growth, `2.723x` 1-record batch throughput), unified SGLang streaming postprocessor builder, and collision-only `pid`/`agent_instance` Prometheus labels.
