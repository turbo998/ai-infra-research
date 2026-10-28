# Phase 126 Track B: SGLang KV Lifecycle & Allocator-Boundary Pins, vLLM Sparse MLA 65535 Grid Cap & R3 Init Order, TensorRT-LLM Kimi-K3 Native C++ & MX Input Processor

Deterministic CPU/offline analytical suite (`own-offline`) modeling:
1. **SGLang KV Allocation Lifecycle, Routed Cache & Allocator-Boundary RDMA Pins (`#46988`, `#46987`, `#46985`, `#46984`, `#46983`, `#46981`)**: Separating byte storage (`StaticKVStorage`) from `IdSpaceKind` allocation identities, pinning unified memory pages at allocator boundaries via `disagg_transfers_in_flight` (`0` compaction-vs-RDMA hazards), gating ROCm fused KV writes on SWA sub-pools (`0` window translation corruptions), and forcing `page_first_direct` for `MambaPoolHost`.
2. **vLLM FlashInfer Sparse MLA `gridDim.z <= 65535` Cap, Pre-Profile R3 Capturer & DeepSeek-OCR Patch Guard (`#63791`, `#63237`, `#64091`, `#64063`)**: Chunking `trtllm-gen` sparse MLA launches to `<= 65,535` tokens and capping DCP workspace (`256.0 MiB` saved at `131,072` tokens), allocating R3 routed-expert buffers before KV cache profiling (`0` post-profile OOMs), and replacing `torch.sum(patches) == 0` with `patches.numel() == 0`.
3. **TensorRT-LLM Kimi-K3 Native C++ Consolidation, MX Input Processor & Perf-Sanity Warmup (`#20893`, `#20762`, `#21123`)**: Linking 4 Kimi-K3 native C++ targets (`mamba2_mtp_ssm_cache_src`, `kda_decode_src`, `kimi_k3_attn_res_src`, `kimi_k3`) into stock libraries (`1.274x` decode speedup), enabling registered HF input processors for `"MX"` checkpoints (`100%` hit rate), and reducing measurement CoV (`14.8% -> 1.9%`).

## Run
```bash
python experiments/sglang_kv_lifecycle_byte_storage_pins_vllm_sparse_mla_65535_r3_trtllm_kimi_k3_phase126/run.py
python experiments/sglang_kv_lifecycle_byte_storage_pins_vllm_sparse_mla_65535_r3_trtllm_kimi_k3_phase126/run.py --check
```
