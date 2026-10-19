# DeepEP V2.5 `HybridElasticSymmetricMemory` & `Engram` GPU-Initiated RDMA Fetch

Deterministic stdlib model of `deepseek-ai/DeepEP` (`commit b5a33eefda39485f438dc764695b71ba7b9249b6`):
- `csrc/kernels/comm/symmetric.hpp` (`HybridElasticSymmetricMemory`): Contiguous VA mapping of GPU symmetric memory (`4 GiB`) + intra-node CPU NUMA RDMA storage shards (`8 × 16 GiB = 128 GiB/node`, `1.0 TiB` across `64` GPUs) via `pidfd_open`/`pidfd_getfd` + `cuMemImportFromShareableHandle`, saving `16.0 GiB/GPU` (`1,024 GiB` cluster HBM).
- `deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh` (`engram_fetch_impl`): Warp-ballot request coalescing (`__ballot_sync` + circular `pending_requests[2*32]` + `ncclGinOptFlagsAggregateRequests`) + atomic last-warp `gin.flush_async`, cutting RDMA doorbells by `10.67x` (`1,792 -> 168`) and reducing 4-layer Engram lookup latency from `676.08 us` (CPU proxy) / `305.68 us` (uncoalesced GPU IBGDA) to `33.56 us` (`20.15x` / `9.11x` speedup).
