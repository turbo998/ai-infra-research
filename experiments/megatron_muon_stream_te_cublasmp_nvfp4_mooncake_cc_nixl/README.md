# Megatron-LM LayerShardedMuon Streaming Newton-Schulz, TE cuBLASMp NVFP4 Overlap & Mooncake RDMA CC / NIXL Trace

Deterministic Python 3.12 standard-library model reproducing:
1. **Megatron-LM PR #8379 (`6c956376e3`)**: Streams local Newton-Schulz chunks via `_iter_ns(full_by_k, load=ns_input)` and in-place `orth.mul_(scale).mul_(self.extra_scale_factor)` in `LayerShardedMuon`, cutting single-rank MoE domain NS launches from `17` to `5` (`3.4x`) and reducing peak transient FP32 memory from `1,008 MiB` to `192 MiB` (`5.25x`).
2. **TransformerEngine PR #3634 (`6b5cd7e2e4`)**: Adds NVFP4 (`VEC16_UE4M3`) `cuBLASMp` (`>= 0.11.0`) AllGather+GEMM and GEMM+ReduceScatter overlap using a 12-byte device scalar buffer `{alpha, 0.0f, 1.0f}` (`CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_POINTER_MODE`), 128-column per-rank split alignment fallback, and single dual-usage (`rowwise` + `columnwise`) quantization of `grad_output`.
3. **Mooncake PR #4197 (`3b08b77168`) & NIXL PR #2462 (`f8248873bd`)**: Mooncake Transfer Engine shared adaptive RDMA congestion control core (`[4 MiB, 64 MiB]` AIMD window, `2000 us` target drain time, generation-fenced `Permit`) and NIXL `nixl::xfer.complete` trace coverage across all 4 completion paths (`50/50` vs `33/50` on UCX).

## Run

```bash
python experiments/megatron_muon_stream_te_cublasmp_nvfp4_mooncake_cc_nixl/run.py --check
```
