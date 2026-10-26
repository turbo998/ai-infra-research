# SGLang CuTe-DSL Multi-Chunk Norm, One-Pass Spatial-Parallel Causal Conv3D & GLM Streaming Parser, vLLM Persistent TopK Exact Overflow Recovery, ModelOpt NVFP4 Dummy Scales, ROCm AITER W4A16 EP & MRV2 Warm-Up Sampler Profiling, and TensorRT-LLM Native UPB Protobuf / Canonical CuTeDSL MoE Identities Deep Dive (Phase 89)

> **中文版**: [sglang-diffusion-vllm-topk-nvfp4-kv-offload-trtllm-grpc-phase89.md](sglang-diffusion-vllm-topk-nvfp4-kv-offload-trtllm-grpc-phase89.md)  
> **Evidence Class**: `implementation` (pinned commit source review) + `author-benchmark` (upstream PR tests and measurements) + `own-offline` (`experiments/sglang_diffusion_vllm_topk_nvfp4_kv_offload_trtllm_grpc_phase89/run.py` deterministic stdlib model)

---

## 1. Architectural Context & Problem Statement

Across multimodal generation (Diffusion VAE/DiT), sparse long-context attention (DSA TopK), and high-throughput MoE serving engines (SGLang, vLLM, TensorRT-LLM), production reliability and tail latency are frequently compromised by **non-deterministic reduction order across warp chunking tactics, $O(N^2)$ streaming regex rescans on unclosed tool calls, histogram bucket candidate overflow in TopK kernels, underestimated sampler peak memory during warm-up profiling, and Python protobuf serialization overhead in gRPC frontends**. This document analyzes 10 non-overlapping commits merged on `2026-10-26` across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`.

---

## 2. Engine Implementation & Failure-Isolation Deep Dive

### 2.1 SGLang Diffusion & Function-Call Parser (PR #46227, #46174, #45143)
1. **Bitwise-Invariant Multi-Chunk CuTe-DSL Norm (`sglang@095750baf9`, PR #46227)**:
   - Each 128-bit vector load covers `CHUNK = WARP_SIZE * 8 = 256` columns (`20` chunks per row at `D = 5120`).
   - To allow a single warp to walk multiple chunks without altering floating-point summation order, `cta_reduce_sum` allocates `num_chunks + 1` `Float32` slots in SMEM (instead of `num_warps + 1`), storing each chunk's warp-local sum at `acc[chunk]` before warp 0 shuffle-reduces `acc`. This guarantees **bitwise-identical BF16 outputs regardless of chunks walked per warp**.
2. **One-Pass Spatial-Parallel Causal Conv3D (`sglang@c4e335f84e`, PR #46174)**:
   - Adds `fill_halo_rows_` and `cat_pad_channels_last_3d` in `parallel_conv.py` to build spatial-parallel causal 3D convolution inputs and halo rows in a single pass instead of two separate memory-bound passes.
3. **O(1) Per-Chunk Tail Scan for Unclosed GLM Tool Calls (`sglang@316ff4dce2`, PR #45143)**:
   - Tracks `self._open_call = (name, args_start)` and `self._end_scan_pos` in `Glm4MoeDetector` and `Glm47MoeDetector` so streaming chunks only scan the newly appended tail for `</tool_call>|$`, cutting regex scan work by **`500.5x`** over 1,000 chunks.

### 2.2 vLLM Persistent TopK Overflow Recovery, NVFP4 Dummy Scales, ROCm AITER W4A16 EP, MRV2 Sampler Profiling & H200 FP8 MoE (PR #62936, #63698, #59255, #57926, #62763)
1. **Exact Persistent TopK on Candidate Overflow & Sliced-View Alignment Guard (`vllm@33a73daf17`, PR #62936)**:
   - Guards `vec_size = 4` behind `stride % 4 == 0 && base % 16 == 0` in `csrc/topk.cu` so sliced views like `x[:, 1:]` do not fault on unaligned `float4` loads.
   - Moves medium-path shared state into dynamic SMEM (`kMediumHistBytes = 3072 B`) and adds `MediumResume` + bounded-memory multi-round radix rescan (`exact_topk_rescan` with `kExactCandidateOffset = 16 KiB` final-round staging) in `csrc/persistent_topk_overflow.cuh`, guaranteeing exact TopK even under massive `-inf` or tie collisions.
2. **Valid ModelOpt NVFP4 MoE Dummy Scales (`vllm@4f140a23de`, PR #63698)**:
   - Initializes positive dummy scales with matching gate/up global scales (`w13_weight_scale = w2_weight_scale = 1.0`, `w13_weight_scale_2 = w2_weight_scale_2 = 1e-3`, `w13_input_scale = w2_input_scale = 1e-2`) so `--load-format dummy` benchmarks on FlashInfer MegaMoE produce finite logits.
3. **ROCm AITER Triton W4A16 Naive Expert Parallelism (`vllm@536854bf10`, PR #59255)**:
   - Passes `gate_valid = (expert_map[topk_ids.long()] >= 0).to(torch.int32)` so scatter-combine skips off-rank experts instead of zero-filling their output rows on AMD `gfx950` / `gfx1250`.
4. **MRV2 Warm-Up Sampler Profiling (`vllm@7f56f453ed`, PR #57926)**:
   - Runs both the default fast-path sampler pass and a warm-up-configured pass (`top_k / top_p / logprobs`) during `_dummy_sampler_run` memory profiling so KV cache sizing accounts for worst-case sampler workspace memory.
5. **H200 FP8 Fused MoE Config for TP8 (`vllm@61b2527bdb`, PR #62763)**:
   - Adds tuned Triton 3.8.0 tile configs for `E=256, N=320, NVIDIA_H200, fp8_w8a8` (`BLOCK_SIZE_M=16, N=64, K=128`) and supports `per_act_token_quant` in `benchmark_moe.py`.

### 2.3 TensorRT-LLM Native UPB Protobuf & Canonical CuTeDSL MoE Identities (PR #19760, #20929)
1. **Native UPB Protobuf for gRPC (`TensorRT-LLM@6448b1b17f`, PR #19760)**: Sets `ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb` in container images and adds `_warn_if_python_protobuf()` on OpenEngine/SMG gRPC server startup.
2. **Canonical CuTeDSL MoE Identities (`TensorRT-LLM@79d1d922dc`, PR #20929)**: Consolidates Blackwell CuTeDSL MoE implementations under `cutedsl/` with `trtllm.cutedsl.<kernel>.<quant>` identities (e.g., `TrtllmCutedslDenseGemmNvfp4Impl`) enforced by registry unit tests.

---

## 3. Cross-Architecture Comparison (Fireworks / Together / Cerebras / Groq)

| Dimension | SGLang / vLLM / TensorRT-LLM (Phase 89) | Fireworks AI / Together AI | Cerebras WSE-3 / Groq LPU |
| :--- | :--- | :--- | :--- |
| **TopK & Sampler Memory Safety** | Bounded 16 KiB radix rescan + dual-pass MRV2 warm-up sampler profiling prevent accuracy collapse and OOM | Multi-tenant APIs must reserve workspace headroom for dynamic `top_k/top_p/logprobs` requests | Host/dedicated post-processing avoids competing with on-chip weight SRAM |
| **Control-Plane Serialization** | Incremental O(1) tool-call tail scan + native `upb` protobuf eliminate Python CPU bottlenecks | Critical for high-concurrency agentic tool-call streaming | Sub-2 ms token generation makes host gRPC serialization an immediate bottleneck if left in pure Python |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Pointer Alignment Exercise**: Explain why a tensor slice `x[:, 1:]` on an FP32 matrix with `stride = 128256` satisfies `stride % 4 == 0` yet faults on `float4` (`16`-byte aligned) vectorized loads unless `base % 16 == 0` is also checked.
2. **Falsifiable Follow-Up**: Measure the exact GPU peak workspace delta between single-pass and dual-pass `_dummy_sampler_run` profiling at `batch_size = 256, vocab_size = 152064` with `top_k = 50, top_p = 0.95, logprobs = 20`.
