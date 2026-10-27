# TransformerEngine NVRTC NVFP4 `4over6` Quantization & Fused EP Dispatch, Megatron-LM `DBuffer` Direct P2P & Batch-Invariant Mamba Prefix Caching, Mooncake Borrowed-Buffer COO Sparse Planner, and NIXL GPU-to-CPU Proxy Ring (Phase 105)

> **Language / 语言**: [中文 (Chinese)](te-nvrtc-nvfp4-ep-fused-megatron-dbuffer-mamba-mooncake-nixl-phase105.md) | **English (英文版)**

## 1. Architectural Context & Core Bottlenecks

Across NVFP4 mixed-precision training, CUDA-graphed MoE Expert Parallelism (EP), MFSDP v2 distributed buffers, deterministic Mamba inference, and sparse RL weight refit / GPU-initiated storage transfers, five infrastructure bottlenecks arise:
1. **Static NVFP4 `4over6` Template Bloat & Host Sync Stalls in Non-Eager EP Dispatch**: Compiling all 24 static NVFP4 `4over6` template instantiations inflates build time, while unfused EP `prepare + dispatch` requires a host `cudaStreamSynchronize` (`46.0 us/layer`) incompatible with CUDA graph capture.
2. **MFSDP v2 Intermediate Staging Buffers & Disabled Mamba Prefix Caching under `batch_invariant_mode`**: MFSDP v2 allocated **`256.0 MiB`** of staging memory during gather/scatter, while Megatron-LM asserted `not self.enable_prefix_caching` whenever `batch_invariant_mode=True`.
3. **Whole-Object Copies & Dense Transfers for Sparse Weight Deltas in Mooncake**: Reading member byte ranges or transferring 12.5%-dense RL/LoRA deltas as full `64.0 MiB` dense tensors wastes over 85% of RDMA fabric bandwidth.
4. **Warp-Wide Ring Contention in GPU-Initiated CPU Proxy Calls**: Without leader election, all 32 lanes in a warp contend on the host proxy ring producer index.

## 2. Primary Source Implementation & Mathematical Invariants

- **TransformerEngine NVRTC NVFP4 `4over6` (`ec39dcf368`) & Fused EP Prepare+Dispatch (`f7273d7d80`)**: Compiles `quantize_4over6_kernel.cuh` at runtime via architecture-aware NVRTC (`24 -> 0` default static AOT instantiations) and fuses CUDA-graph-capturable EP `prepare+dispatch` with device-resident `recv` counts (**`1 -> 0` host syncs/layer, `46.0 -> 21.5 us`, `2.1395x` speedup**).
- **Megatron-LM `DBuffer` Direct P2P (`a666bd60ef`), Batch-Invariant Mamba Prefix Cache (`3888b34882`) & TorchTitan Per-Kernel Error (`efdde5edaf`)**: Uses `DBuffer` owner APIs for direct P2P receives (**saving `256.0 MiB` staging HBM and speeding up gather/scatter `4.80 -> 3.10 ms`, `1.5484x`**); enables batch-invariant Mamba prefix caching when `block_size_tokens % ssm_chunk_alignment == 0` (**cutting 75%-cached 4,096-token prefill to `1,024` tokens, `4.0x` reduction with `0.0` bitwise drift**); and reports per-kernel missing backends in Attention Gym.
- **Mooncake Borrowed-Buffer Byte Ranges & COO Sparse Planner (`81b3694b1b`), LMCache `KVEventSubscriber` (`a0d918a77c`, `6993f46c91`), Dynamo LPX Sidecar (`9ea06e0cd3`) & NIXL 64B CPU Proxy Ring (`f1de8fc286`)**: Reads cross-chunk byte intervals directly into borrowed buffers (`0` intermediate allocs) and plans COO sparse deltas (**`64.0 -> 8.25 MiB`, `7.7576x` bandwidth reduction**); NIXL routes `PROXY = 2` views through a 64-byte ring command written by a single warp leader (**`32.0x` doorbell contention reduction**).

## 3. Deterministic Verification Summary

Run `python experiments/te_nvrtc_nvfp4_ep_fused_megatron_dbuffer_mooncake_phase105/run.py --check`:

| Metric | Baseline | Phase 105 | Improvement | Evidence Class |
|---|---:|---:|---:|---|
| TE NVFP4 `4over6` Default Static Instantiations | `24` | `0` (NVRTC on demand) | **`-100.0%` static build bloat** | `implementation` + `own-offline` |
| TE Fused EP `prepare+dispatch` Latency | `46.0 us` (`1` sync) | `21.5 us` (`0` sync) | **`2.1395x`** | `implementation` + `own-offline` |
| Megatron MFSDP v2 `DBuffer` P2P / Staging HBM | `4.80 ms` / `256.0 MiB` | `3.10 ms` / `0.0 MiB` | **`1.5484x` / `256.0 MiB` saved** | `implementation` + `own-offline` |
| Batch-Invariant Mamba 75%-Cached Prefill | `4,096` tokens | `1,024` tokens | **`4.0x` (`0.0` bitwise drift)** | `implementation` + `own-offline` |
| Mooncake 12.5%-Dense COO Sparse Transfer | `64.0 MiB` | `8.25 MiB` (`0` temp allocs) | **`7.7576x` bandwidth reduction** | `implementation` + `own-offline` |
| NIXL GPU-to-CPU 64B Proxy Ring Warp Producers | `32` threads | `1` leader thread | **`32.0x` contention reduction** | `implementation` + `own-offline` |
