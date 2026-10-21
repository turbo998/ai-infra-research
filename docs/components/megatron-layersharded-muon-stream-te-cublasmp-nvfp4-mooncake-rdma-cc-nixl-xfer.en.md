# Deep Dive: Megatron-LM LayerShardedMuon Streaming Newton-Schulz, TE cuBLASMp NVFP4 Comm+GEMM Overlap & Mooncake RDMA CC / NIXL Trace

> **中文母稿**: [megatron-layersharded-muon-stream-te-cublasmp-nvfp4-mooncake-rdma-cc-nixl-xfer.md](megatron-layersharded-muon-stream-te-cublasmp-nvfp4-mooncake-rdma-cc-nixl-xfer.md)  
> **Phase 41 (`priority: 50`) · Deterministic Experiment**: [`experiments/megatron_muon_stream_te_cublasmp_nvfp4_mooncake_cc_nixl/run.py`](../../experiments/megatron_muon_stream_te_cublasmp_nvfp4_mooncake_cc_nixl/run.py)  
> **7-Card 4:5 Visual Report**: [`docs/reports/ai-infra-phase41-benchmark-cards.html`](../reports/ai-infra-phase41-benchmark-cards.html)

---

## 1. Executive Summary

1. **Megatron-LM `LayerShardedMuon` Streaming Batched Newton-Schulz (`PR #8379` / `6c956376e3`)**:
   - Refactors `LayerShardedMuon._run_ns` into a streaming generator `_iter_ns(full_by_k, load=ns_input)` with in-place post-scaling `orth.mul_(scale).mul_(self.extra_scale_factor)` and enables local shape-batched Newton-Schulz on single-rank `(GTP_remat x TP)` domains under `--use-layer-wise-distributed-optimizer`.
   - For a typical rank holding `1` dense matrix (`4096x4096`) and `8` local experts (`16` matrices of `4096x1536` / `1536x4096`, `17` matrices total), `ns_batch_size=4` cuts Newton-Schulz launches from **`17` to `5` (`3.4x`)** and slashes peak transient FP32 memory from **`1,088.0 MiB` to `192.0 MiB` (`5.6667x`)**, bounding live Nesterov `lerp` temporaries per NS call to `<= ns_batch_size + 1` (`5`).
2. **TransformerEngine `cuBLASMp` (`>= 0.11.0`) NVFP4 Comm+GEMM Overlap (`PR #3634` / `6b5cd7e2e4`)**:
   - Enables `nvte_all_gather_gemm` and `nvte_gemm_reduce_scatter` for NVFP4 (`VEC16_UE4M3`, TN layout) using a **`12 B`** device scalar buffer `{alpha, 0.0f, 1.0f}` (`CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_POINTER_MODE`) computed on-stream via `nvte_nvfp4_compute_per_tensor_scale`.
   - Automatically falls back to `CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT` when `(N / tp_size) % 128 != 0`, and quantizes `grad_output` once before `dgrad` with both `rowwise` and `columnwise` usage (`2 -> 1` quantization call, saving `16.0 MiB` BF16 read traffic and preserving stochastic-rounding parity between `dgrad` and `wgrad`).
3. **Mooncake Adaptive RDMA Congestion Control (`PR #4197` / `3b08b77168`) & NIXL Completion Trace Marks (`PR #2462` / `f8248873bd`)**:
   - Mooncake Transfer Engine introduces `adaptive_congestion_control` (`[4 MiB, 64 MiB]` window, `2000 us` target drain time, halving after `2` high-pressure epochs and growing `+1 MiB` after `3` low-pressure epochs, indivisible oversized slice admission on empty domains, and generation-fenced `Permit`).
   - NIXL emits `nixl::xfer.complete` across all 4 completion observation sites (`postXferReq` immediate completion, `getXferStatus`, repost check in `postXferReq`, and `releaseXferReq`), raising UCX completion trace coverage from **`33/50` (`66%`) to `50/50` (`100%`)**.

---

## 2. First-Hand Upstream Commits

| Repository | Commit / PR | Date | Core Files | Evidence ID |
|---|---|---|---|---|
| `NVIDIA/Megatron-LM` | [`6c956376e3` (`#8379`)](https://github.com/NVIDIA/Megatron-LM/commit/6c956376e3ef22528413ff35412ad8b938272986) | `2026-10-21` | `megatron/core/optimizer/layer_sharded_muon.py` | `EV-2026-361` |
| `NVIDIA/TransformerEngine` | [`6b5cd7e2e4` (`#3634`)](https://github.com/NVIDIA/TransformerEngine/commit/6b5cd7e2e43cee6da5dff9b9f86bc2ec05d7b1ad) | `2026-10-21` | `transformer_engine/common/comm_gemm/comm_gemm.cpp`, `linear.py` | `EV-2026-362` |
| `kvcache-ai/Mooncake` & `ai-dynamo/nixl` | [`3b08b77168` (`#4197`)](https://github.com/kvcache-ai/Mooncake/commit/3b08b77168870d265e32b6cfeab661f037426b9c) / [`f8248873bd` (`#2462`)](https://github.com/ai-dynamo/nixl/commit/f8248873bd5b8d3609ba1448fa2f623aefb361cf) | `2026-10-21` | `adaptive_congestion_control.cpp`, `nixl_Agent.cpp` | `EV-2026-363` |

---

## 3. Deterministic Reproduction & Validation Metrics

| Subsystem | Metric | Deterministic Value | Status |
|---|---|---|---|
| Megatron Muon | Peak Transient FP32 Memory (`Eager -> Streaming`, 17 matrices) | `1088.0 MiB -> 192.0 MiB` (`5.6667x` reduction) | `PASS` |
| Megatron Muon | Newton-Schulz Kernel Launches / Max Live Nesterov Lerps | `17 -> 5` (`3.4x` reduction) / `<= 5` | `PASS` |
| TE cuBLASMp NVFP4 | Device Scalar Buffer / `(N/tp)%128!=0` Algorithm Fallback | `12 B` / `SPLIT_P2P_OVERLAP -> DEFAULT` | `PASS` |
| TE cuBLASMp NVFP4 | Backward `grad_output` Quantization Calls (`Before -> After`) | `2 -> 1` (saves `16.0 MiB` read, preserves SR parity) | `PASS` |
| Mooncake RDMA CC | Empty-Domain `8 MiB` Slice / Window Trajectory (`12 low + 2 high`) | `kAllow` / `4 MiB -> 8 MiB -> 4 MiB (kCongested)` | `PASS` |
| NIXL Trace Marks | `nixl::xfer.complete` Marks Emitted Across 50 UCX Transfers | `33/50 -> 50/50 (100%)` | `PASS` |

---

## 4. Production Adoption Recommendations

1. **Streamed Batched Muon for MoE**: Configure `--muon-ns-batch-size 4` with `--use-layer-wise-distributed-optimizer` so single-rank expert domains execute batched 3D SYRK Newton-Schulz with `5.67x` lower transient FP32 memory.
2. **Enforce 128-Column Per-Rank Alignment for NVFP4 Overlap**: Ensure `(N / tp_size) % 128 == 0` on TransformerEngine linear layers to stay on the `SPLIT_P2P_OVERLAP` path in `cuBLASMp >= 0.11.0`.
