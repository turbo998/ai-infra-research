# Megatron-LM MFSDP v2 Unpadded Shards & HyperConnection Residual Contracts, NCCL GIN Socket GDRCopy Guard & nccl4py Completion Event, TransformerEngine Forward Fusion Compile, FA4 SM100 Block-Sparse Empty-Tile Zero-Fill, Mooncake DMA-BUF Page Alignment, LMCache SGLang Salt Isolation, Dynamo GPT-OSS Channels & NIXL Mooncake postXfer Leak Fix (Phase 121)

> **Module Positioning**: Written for cloud and AI infrastructure solution architects designing distributed training frameworks (Megatron-LM, TransformerEngine, FlashAttention-4, NCCL) and disaggregated KV transport engines (Mooncake, LMCache, Dynamo, NIXL). Based on 10 primary upstream commits landed between 2026-09-17 and 2026-10-28.
> **Evidence Scope**: `implementation` (verified against pinned commit diffs) + `own-offline` (reproducible Python standard-library modeling in this repository).

---

## 1. Architectural Overview & 10 Upstream Commits

| # | Repository & Commit | Date | Core Mechanism & Architectural Change | Evidence Scope |
| :--- | :--- | :--- | :--- | :--- |
| C1 | `NVIDIA/Megatron-LM` [`47715bb`](https://github.com/NVIDIA/Megatron-LM/commit/47715bb029f59aef4c45d516379911729e6b375d) | 2026-10-27 | `megatron_fsdp`: Add `ShardLayout.UNPADDED` (`megatron_fsdp_padded_shards=False`) for `RowAtomic`/`BlockAtomic` groups using uneven collectives | `implementation` |
| C2 | `NVIDIA/Megatron-LM` [`31c7230`](https://github.com/NVIDIA/Megatron-LM/commit/31c723006edcc345c89416959f24b94b3d34eded) | 2026-10-27 | `transformer`: Extract `HyperConnectionTransformerLayer` ($H_{\text{pre}}, H_{\text{post}}, H_{\text{res}}$) and `WideResidual` / `residual_recompute` modules | `implementation` |
| C3 | `NVIDIA/nccl` [`d8347eb`](https://github.com/NVIDIA/nccl/commit/d8347eb9b864fab0e79e58c4ca1b9eba5618a136) | 2026-09-17 | `gin_host_socket.cc`: Remove lazy `ncclGdrInit()` in `ncclRmaSocketProxyInit` so `NCCL_GDRCOPY_ENABLE=0` fails fast with `ncclInternalError` | `implementation` |
| C4 | `NVIDIA/nccl` [`373bb82`](https://github.com/NVIDIA/nccl/commit/373bb82d1f08be1227e896dae02a1d3f410c97cd) | 2026-09-17 | `nccl4py`: Expose `NCCLCollConfig.launch_completion_event` (`NcclEventSpec`, rejecting `0` or `bool`) | `implementation` |
| C5 | `NVIDIA/TransformerEngine` [`45223d7`](https://github.com/NVIDIA/TransformerEngine/commit/45223d764a28fdaae242de3f0077c34f538d00de) | 2026-10-08 | `fuser.py`: Compile `ForwardLinearBiasActivation`, `BasicLinear` (specialized by TP size), and `Bias` with `torch.compile(fullgraph=True)` | `implementation` |
| C6 | `Dao-AILab/flash-attention` [`86dcb5d`](https://github.com/Dao-AILab/flash-attention/commit/86dcb5d60c4e3bfd8117cb71e8a209afc80edcf9) | 2026-09-24 | `flash_fwd_sm100.py`: Pass `zero_fill=True` in `correction_epilogue` for empty block-sparse Q tiles to prevent `0.0 * Inf = NaN` | `implementation` |
| C7 | `kvcache-ai/Mooncake` [`058c1e8`](https://github.com/kvcache-ai/Mooncake/commit/058c1e8a0e872caf4c9175c737bba7ed60525e55) | 2026-10-28 | `rdma_context.cpp`: Page-align `exportSize` for whole allocations in `RdmaContext::exportDmabuf` | `implementation` |
| C8 | `LMCache/LMCache` [`577cca9`](https://github.com/LMCache/LMCache/commit/577cca98ca22bc75416b085523217b4bcff3cd81) | 2026-10-22 | `test_unified_sglang_mp_connector.py`: Verify `cache_salt` isolation across `submit_lookup`, `_create_key`, and `ipc_key_to_object_keys` | `implementation` |
| C9 | `ai-dynamo/dynamo` [`532616d`](https://github.com/ai-dynamo/dynamo/commit/532616d12ddea4d9ebce30fbcb67807cc02b63e0) | 2026-10-28 | `http/service.rs`: Keep `reasoning_disabled = false` for `gpt_oss` with `parsers 0.7.35` and serialize tracing callsite interest rebuilds | `implementation` |
| C10 | `ai-dynamo/nixl` [`7bd1d13`](https://github.com/ai-dynamo/nixl/commit/7bd1d1386ecb5c24ce51f16c9070d25dd93492e4) | 2026-10-12 | `mooncake_backend.cpp`: Replace `new[]`/`delete[]` with `std::vector<transfer_request_t>` in `postXfer` to fix early-return leak | `implementation` |

---

## 2. Deep Technical Analysis & Formulations

### 2.1 Megatron-FSDP v2 `ShardLayout.UNPADDED` for Heterogeneous Row Widths (`47715bb`)

- Under `ShardLayout.PADDED`, `build_for_row_atomic` aligns every tensor and rank boundary to $\text{chunk\_size} = \operatorname{lcm}_i(\text{block\_size} \cdot \text{row\_size}_i)$. For mixed parameter shapes `[(128, 6144), (128, 4096), (128, 14336)]` with `dp_size = 8, block_size = 32`, the true element count is $3,145,728$, whereas LCM padding inflates the group buffer to **$22,020,096$ elements ($18,874,368$ padded elements wasted, 7.0x inflation)**.
- `ShardLayout.UNPADDED` (`megatron_fsdp_padded_shards=False`) packs tensors contiguously ($3,145,728$ elements, zero padding), splits rank boundaries at row or 32-row block boundaries via `bisect.bisect_left(row_ends, (k * size) // dp_size)`, and dispatches `all_gather_into_tensor_uneven` / `reduce_scatter_tensor_uneven` (while rejecting `nccl_ub=True`).

### 2.2 FA4 SM100 Empty Block-Sparse Tile Zero-Fill & Mooncake DMA-BUF Page Alignment (`86dcb5d`, `058c1e8`, `7bd1d13`)

- **FA4 SM100 Empty Tile `0.0 * Inf = NaN` Fix (`86dcb5d`)**: When a block-sparse Q tile has `total_block_cnt == 0`, `scale = +Inf` and reading uninitialized TMEM containing `0.0` produces `0.0 * Inf = NaN`. Passing `zero_fill=True` to `correction_epilogue` skips the TMEM read/rescale and executes `tOrO_frg.fill(0.0)` with LSE set to $-\infty$.
- **Mooncake DMA-BUF Page Alignment (`058c1e8`) & NIXL RAII (`7bd1d13`)**: `RdmaContext::exportDmabuf` rounds whole-allocation export sizes up to OS page alignment ($\text{exportSize} = (\text{allocSize} + \text{page} - 1) \ \& \ \sim(\text{page} - 1)$, turning `4,206,649 B` into `4,210,688 B`), and `nixlMooncakeEngine::postXfer` uses `std::vector<transfer_request_t>` to eliminate an 80-byte leak on length-mismatched 2-descriptor calls.

---

## 3. Reproducible Offline Verification (`own-offline`)

Experiment directory: `experiments/megatron_mfsdp_unpadded_hyperconnection_nccl_gin_te_compile_fa4_mooncake_lmcache_dynamo_nixl_phase121/`

- **Command**: `python experiments/megatron_mfsdp_unpadded_hyperconnection_nccl_gin_te_compile_fa4_mooncake_lmcache_dynamo_nixl_phase121/run.py --check`
- **Verified Sealed Outputs**:
  - `megatron_mfsdp_and_hyperconnection.unpadded_saved_numel = 18874368` (`22,020,096 -> 3,145,728`);
  - `nccl_te_and_fa4.fa4_empty_tile_buggy_is_nan = true`, `fa4_empty_tile_fixed_val = 0.0`;
  - `mooncake_lmcache_dynamo_nixl.mooncake_aligned_export_bytes = 4210688`, `nixl_fixed_leaked_bytes_per_mismatch = 0`.

---

## 4. Architect Exercises & Falsifiable Questions

1. **Calculation Exercise**: On a 64 KiB page system (`page = 65,536 B`), what is the rounded `exportSize` in `RdmaContext::exportDmabuf` for a whole `cudaMalloc` allocation of `10,000,000 B`?
   - *Answer*: $153 \times 65,536 = 10,027,008\text{ Bytes}$.
2. **Falsifiable Question**: What is the NVLink bandwidth tradeoff between uneven `all_gather_into_tensor_uneven` under `ShardLayout.UNPADDED` and symmetric-memory `nccl_ub=True` collectives under `ShardLayout.PADDED`?
