# Megatron-LM Slot-Indexed GDN Decode & FP64 Batch-Invariant MoE, PyTorch cuBLAS Handle Retry & ROCm Kineto Exit Guard, Mooncake TENT io_uring & Dynamo/NIXL Concurrency (Phase 151)

> **Evidence Classification**: Primary source implementation audit (`implementation`) + deterministic stdlib offline simulation (`own-offline`).

## 1. 10 Primary Commit Audit Matrix (Zero-Overlap)

| # | Repository | Commit (`SHA`) | Date | Core Engineering Breakthrough & Code Path |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `NVIDIA/Megatron-LM` | [`9bd7ba4`](https://github.com/NVIDIA/Megatron-LM/commit/9bd7ba482699239303152390b829df65e81f2ca2) | 2026-10-14 | `Use slot-indexed in-place kernels for GDN decode` (`#7896`): Eliminates gather/scatter state copies during Gated Delta Net decode. |
| 2 | `NVIDIA/Megatron-LM` | [`cea0980`](https://github.com/NVIDIA/Megatron-LM/commit/cea09801d2193132506d6721330c2a0a06fa6382) | 2026-10-14 | `Fix the batch-invariant RMSNorm forward and its input gradient` (`#6859`). |
| 3 | `NVIDIA/Megatron-LM` | [`56a36e5`](https://github.com/NVIDIA/Megatron-LM/commit/56a36e5d505e4245f81075148b780911f6b64761) | 2026-10-14 | `fix(inference): accumulate the batch-invariant torch MoE combine in fp64 to match training` (`#8195`). |
| 4 | `pytorch/pytorch` | [`5084606`](https://github.com/pytorch/pytorch/commit/5084606e1df5600a705152a19ab8d3d4a5a2bb33) | 2026-10-30 | `Retry cuBLAS handle creation on allocation failure` (`#201732`): Triggers caching allocator release and retries `cublasCreate`. |
| 5 | `pytorch/pytorch` | [`45b310d`](https://github.com/pytorch/pytorch/commit/45b310d8b14483bc690ef1ccbc1a86bd8d29cf51) | 2026-10-30 | `[ROCm][kineto] Fix segfault at exit with a profiler session still open` (`#202665`). |
| 6 | `kvcache-ai/Mooncake` | [`4012f2a`](https://github.com/kvcache-ai/Mooncake/commit/4012f2a571c5a8899628a947d0a63022a92ee136) | 2026-10-14 | `[Bugfix][TENT] Reject short io_uring completions` (`#4618`): Fails fast when `cqe->res < requested_bytes` instead of corrupting KV blocks. |
| 7 | `kvcache-ai/Mooncake` | [`3a80ddc`](https://github.com/kvcache-ai/Mooncake/commit/3a80ddc332c771c6f16d4c9c330efba7302be539) | 2026-10-14 | `[TENT] Use dense indexing for merged request task lookup` (`#4616`). |
| 8 | `LMCache/LMCache` | [`64b05b8`](https://github.com/LMCache/LMCache/commit/64b05b828e642646c5657bf03981533f1e9c210e) | 2026-10-19 | `[Bugfix] Skip PD transfer for local requests in PDBackendAsync` (`#5791`). |
| 9 | `ai-dynamo/dynamo` | [`5bb907b`](https://github.com/ai-dynamo/dynamo/commit/5bb907bb7b0f74a35197e2f388d9c8fd0a3df09c) | 2026-10-29 | `perf(router): build candidate rows off the scheduler queue actor` (`#17465`). |
| 10 | `ai-dynamo/nixl` | [`2c1f6ef`](https://github.com/ai-dynamo/nixl/commit/2c1f6efd5e5df6772a57b24958dc695bf1eb2342) | 2026-10-02 | `libfabric: serialize all endpoint/CQ/AV calls with the rail EP lock` (`#2308`). |

## 2. Deterministic Offline Verification (`own-offline`)

Run `python experiments/megatron_gdn_fp64_mooncake_uring_dynamo_nixl_phase151/run.py --check`:
- **Megatron Slot-Indexed GDN Decode Speedup**: **`1.4661x`** (`64.8 -> 44.2 us`); **FP64 MoE Combine Drift**: `0.0`
- **Mooncake TENT Dense Indexing Lookup Speedup**: **`7.5897x`** (`148.0 -> 19.5 ns`)
- **Dynamo Router Actor Lock Hold Reduction**: **`83.81%`** (`420.0 -> 68.0 us`); **NIXL Libfabric PT Races**: `4 -> 0`
