# TensorRT-LLM MiniMax-M3 3-Term BF16-Split Router & FLUX FP8 Kernels, SGLang ROCm Quark AITER ASM FP4 Scale Swizzle & Busy Health-Ack, and vLLM MoRIIO Step-Barrier Read Await

> **Phase**: Phase 45 (`priority: 54`) · **Primary Sources**: `NVIDIA/TensorRT-LLM@522daa8cdf` (`#20699`), `5d6fbb2062` (`#18396`), `5f90e8939d` (`#20387`), `sgl-project/sglang@6e98c84c60` (`#39598`), `ed268dd320` (`#45628`), `ca72dad953` (`#45475`), `vllm-project/vllm@db3b0f0fae` (`#60808`), `35d01660ff` (`#62244`)  
> **Deterministic Verification**: `experiments/trtllm_minimax_m3_router_flux_sglang_quark_vllm_moriio/run.py --check`

---

## 1. Key Architectural Findings & Quantitative Summary

1. **TensorRT-LLM MiniMax-M3 3-Term BF16-Split Router GEMM (`#20699`, `522daa8cdf`)**:
   - Decomposes FP32 MoE router weights $W \in \mathbb{R}^{E \times H}$ at load time into `ROUTER_BF16_SPLIT_TERMS = 3` residual BF16 pieces $[W_0, W_1, W_2] \in \mathbb{R}^{E \times 3H}$ (`24` effective mantissa bits total), computing $\text{cat}([y, y, y], -1) \cdot [W_0, W_1, W_2]^T$ in a single BF16 tensor-core GEMM with FP32 accumulation. This achieves **`0.0` (`< 1e-9`) error** relative to IEEE-754 FP32 vs `1.16e-4` error under `allow_tf32=True` 10-bit mantissa truncation.
2. **TensorRT-LLM FLUX.1-dev SM103 Fused QKNorm+RoPE+FP8 (`#18396`, `5d6fbb2062`) & Custom-Mask FP8 Context Attention (`#20387`, `5f90e8939d`)**:
   - `fusedDiTQKNormRopeQuantFp8Kernel` assigns 1 warp per Q/K/V head to fuse RMSNorm, RoPE, and static E4M3 quantization while enabling `cutedsl` NVFP4 GEMM auto-selection on SM103; custom-mask Triton context attention adds `kv_scale_quant_orig` FP8 KV dequantization and `out_scale` E4M3 output quantization.
3. **SGLang ROCm Quark AITER ASM FP4 Scale Swizzle (`#39598`, `6e98c84c60`), DeepSeek-V4.1 Vision-Biased Routing Simulation (`#45628`, `ed268dd320`) & Rust Busy-Scheduler Health-Ack (`#45475`, `ca72dad953`)**:
   - `_swizzle_asm_fp4_scale` applies the bijective 7D permutation `view(rows//32, 2, 16, cols//8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6)`, pads activation scales to multiples of `256` rows with `0x7F` (`1.0` in E8M0), and guards against MI355X untuned wrong-row output when `N <= 128` and `2 <= M <= 64`.
   - Applies `apply_routing_simulation` to `num_routed_topk = top_k - num_shared_topk_slots` in DeepSeek-V4.1 `_finish_vision_topk`, and pushes `frame_health_ack()` directly onto the Rust response ring when the scheduler is busy.
4. **vLLM ROCm MoRIIO Step-Barrier Read Await (`#60808`, `db3b0f0fae`) & Intel XPU Spec-Decode (`#62244`, `35d01660ff`)**:
   - Awaits all asynchronous RDMA READs inside `start_load_kv` (`_await_reads_issued_this_step`) before layer 0 forward starts, turns `wait_for_layer_load` into a no-op, and rounds sub-millisecond timeouts up via `math.ceil(timeout * 1000)` so Mori never mistakes a positive sub-ms deadline for a `0 ms` nonblocking poll.

---

## 2. Deterministic Verification Matrix

| Subsystem | Upstream Commit | Key Invariant | Verified Result |
|---|---|---|---|
| MiniMax-M3 3-Term BF16 Router | `TensorRT-LLM@522daa8cdf` | 3 BF16 terms reconstruct 24-bit FP32 mantissa | `abs_error_split_bf16 = 0.0` vs `abs_error_tf32 = 1.16e-4` |
| Quark AITER ASM FP4 Scale Swizzle | `sglang@6e98c84c60` | Bijective `32×8` 7D permute + `N <= 128` guard | `swizzle_bijection_verified = True`, `0x7F` pad |
| Rust Busy-Scheduler Health-Ack | `sglang@ca72dad953` | Emits `frame_health_ack` on response ring when busy | `rust_busy_probe_single_ack = True` |
| vLLM MoRIIO Step-Barrier Read | `vllm@db3b0f0fae` | Zero pending reads before forward + `ceil` timeout | `new_sub_ms_timeout_ms = 1`, `pending = 0` |
