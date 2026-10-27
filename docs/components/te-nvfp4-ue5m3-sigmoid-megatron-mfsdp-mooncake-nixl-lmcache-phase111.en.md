# Phase 111 Track C: TransformerEngine NVFP4-UE5M3 & Biased Sigmoid Score Preservation, Megatron-LM MFSDP v2 & Multi-Axis QuantizedDBuffer, verl Sticky DP Pinning, and Mooncake/NIXL/LMCache Transfers

> **Provenance & Evidence Boundary**: Derived from first-party public GitHub commits across TransformerEngine, Megatron-LM, verl, Mooncake, NIXL, and LMCache plus our deterministic stdlib simulation (`experiments/te_nvfp4_ue5m3_megatron_mfsdp_mooncake_nixl_lmcache_phase111/run.py`).

---

## 1. TransformerEngine: NVFP4 with FP8 `UE5M3` Block Scales, Biased Sigmoid Score Preservation & `no_grad` UB Overlap

### 1.1 Prototype NVFP4 with FP8 `UE5M3` Block Scales (`NVIDIA/TransformerEngine@d0b4b32c8e90`, `#3325`)
- Adds `DType::kFloat8UE5M3` (`__nv_fp8_ue5m3`, CUDA >= 13.4) block scales for NVFP4 (`float4e2m1`), routing 256-aligned dense GEMMs and `wgrad` through cuDNN's fused `e5m3` kernels and falling back to dense/general `wgrad` kernels for unaligned or grouped shapes.

### 1.2 Preserving Tiny Sigmoid Scores in Biased Fused Routing (`5e9946406256`, `#3637`) & `no_grad` UB Overlap (`6eb41bd20f16`, `#3546`)
- **Tiny Sigmoid Score Preservation (`5e9946406256`)**: In `fused_topk_with_score_function.cu`, adding `expert_bias` (e.g. `0.75`) to a tiny sigmoid score ($\sigma(-18) \approx 1.523\times 10^{-8}$) rounds away the score in FP32 so post-top-k subtraction returns `0.0`. For `ScoreFunc == 0` (sigmoid), the kernel now gathers selected scores directly from the saved unbiased `intermediate_output` buffer.
- **`LayerNormLinear` UB AG Overlap Under `torch.no_grad()` (`6eb41bd20f16`)**: Drops the `is_grad_enabled` gate on `ub_overlap_ag_fprop` in `layernorm_linear.py`.

---

## 2. Megatron-LM MFSDP v2 Checkpointing & Multi-Axis `QuantizedDBuffer`, plus verl Sticky DP Session Pinning

- **MFSDP v2 Expert-Parallel MoE Checkpointing (`NVIDIA/Megatron-LM@db3a6cd8a03e`, `#8230`)**: Calls `sync_model_weights_from_main_weights` inside `FullyShardedDataParallel.load_state_dict` so BF16/quantized compute buffers are refreshed from DCP-loaded FP32 main weights before the first resumed forward pass, and retains empty local shards in `FullyShardedOptimizer._param_to_fqn` on TE `< 2.18.0`.
- **Multi-Axis `QuantizedDBuffer.allgather` (`e6913b8f3225`, `#7727`)**: Materializes `mesh_axis` into a `tuple` once before iterating across planes so generator inputs are not exhausted after the first plane.
- **verl Sticky Multi-Turn vLLM DP Pinning (`volcengine/verl@9058412fb308`, `#8179`)**: Forwards `session_id = request_id` and routes turns to `crc32(session_id) % data_parallel_size` across prefill and decode servers, lifting Nemotron-H 30B-A3B SWE-bench Pro replay prefix-cache hit rate from `0.81` to `0.86` (`0.93` cumulative in full RL runs) and cutting long-session runtime by `10-15%`.

---

## 3. Mooncake Direct NVLink Reads & Striped `ObjectRoute`, NIXL Sub-Range cuObject Tokens & LMCache 2-D Batch Memcpy

- **Mooncake Direct GPU Reads (`kvcache-ai/Mooncake@9c0528d9b486`, `#3431`) & Striped `ObjectRoute` (`f8c900715e23`, `#4585`)**: `RealClient::can_use_direct_memory_read` allows direct reads into ordinary unregistered CUDA device allocations when `replica_protocol == "nvlink"` while requiring registered bounds for `"rdma"`. `ObjectRoute` moves locks onto persistent `KeySlot` entries with per-publication `Generation` IDs.
- **NIXL Sub-Range `cuObject` RDMA Tokens (`ai-dynamo/nixl@a88b4ddc604a`, `#2369`)**: `SharedCuObjClient::getToken` resolves the enclosing registration `[base, base + reg_size)` for any sub-range `[ptr, ptr + size)` and calls `cuMemObjGetRDMAToken(base, size, ptr - base, op)`.
- **LMCache 2-D Direct Copy (`LMCache/LMCache@0d441be069cc`, `#5506`)**: Collapses per-layer 1-D copies into a single 2-D `cudaMemcpy3DBatchAsync` per `(kv_plane, block)` whenever layers in a kernel group sit at a constant byte pitch (vLLM block-outermost layout), cutting copy descriptors by `32x` for a 32-layer group.
