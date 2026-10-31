# Phase 157 Deep Dive: SGLang Page-Envelope & DSA Host Pools, Qwen3/ROCm Fused QK-Norm/RoPE, vLLM SM121/SM103 Skinny GEMM & Direct RL Weight Reload, TRT-LLM MiniMax-M3 NVFP4 Megakernel & Sol-Attn Morton Ordering

> **Language / 语言**: [English](sglang-page-envelope-qwen3-dsa-vllm-skinny-reload-trtllm-minimax-phase157.en.md) | [中文](sglang-page-envelope-qwen3-dsa-vllm-skinny-reload-trtllm-minimax-phase157.md)

## 1. Overview & Architectural Evolution

This report audits 10 primary commits across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM` covering hierarchical KV/DSA host memory pools, fused QK-Norm/RoPE decode kernels, CuTe DSL skinny GEMM plans for `SM121` and `SM103`, in-place RL weight reloading, Blackwell NVFP4 decode megakernels, and 3D Morton Z-order token reordering for video diffusion.

## 2. Primary Source Audit & Quantitative Findings

1. **SGLang Unified Page-Envelope (`dff2c4e`, `#45020`) & DSA Indexer Host Pools (`b44b02e`, `#45019`)**: `PageEnvelopeBufferInfo` registers only backed physical KV pages (cutting pinned DMA registration by **25.0%** on a 64 GiB virtual / 48 GiB physical envelope), while `IndexKeyFormat.DSA_FP8` packs 128 FP8 elements + 1 FP32 scale into `132 B/key` (`8448 B` per 64-key page, a **1.9394x** compression over BF16).
2. **SGLang Qwen3 Decode Fused QK-Norm/RoPE (`8368f86`, `#46908`) & ROCm MiniMax-H3 Bit-Exact RMSNorm (`7585ccc`, `#47569`)**: Replaces 3 separate decode launches with 1 fused kernel (**1.68x** speedup) and replicates ATen's ROCm 32-lane reduction tree for bit-exact BF16/FP32 per-head RMSNorm + RoPE.
3. **vLLM SM121/SM103 Skinny GEMM (`e5196e2`, `#62956`; `900b8b0`, `#64650`), ROCm DSpark Draft Head Quant (`7b77ed5`, `#64368`) & Direct Weight Reload (`b5755c3`, `#64336`)**: CuTe DSL skinny GEMM plans achieve **1.42x** speedup (`13.8 us` vs `19.6 us`) under CUDA Graph replay; DSpark draft LM head quantization cuts draft head memory traffic by **2.0x**; `VLLM_RELOAD_MODE=direct` updates live RL rollout weights in place (**4.50x** faster reload with **0.0%** peak memory overhead).
4. **TensorRT-LLM Blackwell MiniMax-M3 NVFP4 Megakernel (`f405e10`, `#20306`) & Sol-Attn 3D Morton Z-Order (`6adeccc`, `#20787`)**: Cuts MiniMax-M3 decode attention latency to `84.0 us` (**1.56x** speedup) and improves Wan 3D block-sparse spatial locality hit rate by **+24.6 pts** (`86.1%` vs `61.5%`).

## 3. Reproducible Verification

Run `python experiments/sglang_page_envelope_qwen3_dsa_vllm_skinny_reload_trtllm_phase157/run.py --check`.
