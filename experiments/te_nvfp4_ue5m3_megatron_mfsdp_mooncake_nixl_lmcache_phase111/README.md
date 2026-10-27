# Phase 111 Track C Experiment: TE NVFP4-UE5M3 & Biased Sigmoid Router, Megatron MFSDP v2, verl Sticky DP Pinning, Mooncake, NIXL & LMCache 2-D Direct Copy

Deterministic Python standard-library model covering:
1. **TransformerEngine (`d0b4b32c8e90`, `5e9946406256`, `6eb41bd20f16`)**:
   - NVFP4 with FP8 `UE5M3` block scales and 256-aligned cuDNN fused GEMM/wgrad routing.
   - Biased fused routing preserving tiny sigmoid scores via saved unbiased intermediates instead of post-top-k bias subtraction, plus `LayerNormLinear` Userbuffers AG overlap under `torch.no_grad()`.
2. **Megatron-LM (`db3a6cd8a03e`, `e6913b8f3225`) & verl (`9058412fb308`)**:
   - MFSDP v2 expert-parallel MoE checkpoint `sync_model_weights_from_main_weights` and `QuantizedDBuffer` multi-axis `allgather`.
   - verl sticky multi-turn session pinning via `crc32(session_id) % data_parallel_size` across vLLM DP ranks (`0.81 -> 0.86` replay hit rate, `0.93` full RL run).
3. **Mooncake (`9c0528d9b486`, `f8c900715e23`), NIXL (`a88b4ddc604a`), and LMCache (`0d441be069cc`)**:
   - Mooncake Store direct memory read for NVLink ordinary device allocations vs RDMA registered buffers, and striped `ObjectRoute` `KeySlot` + `Generation` concurrency.
   - NIXL `cuMemObjGetRDMAToken(base, size, ptr - base, op)` sub-range RDMA token minting and LMCache 2-D `cudaMemcpy3DBatchAsync` for block-outermost KV layouts (`32x` descriptor reduction for 32 layers).

## Run
```bash
python experiments/te_nvfp4_ue5m3_megatron_mfsdp_mooncake_nixl_lmcache_phase111/run.py --check
```
