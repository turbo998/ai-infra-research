# Phase 156 Deep Dive: SGLang PP Prefill KV Consensus, MiniMax-M3.1 DSpark, vLLM HiSparse Fused Invalidation, Muse Spark Hybrid MoE & TensorRT-LLM Qwen3.8 / One-Model SpecDecode

> **Language / 语言**: [English](sglang-pp-kv-minimax-glm5-vllm-hisparse-muse-trtllm-qwen38-phase156.en.md) | [中文](sglang-pp-kv-minimax-glm5-vllm-hisparse-muse-trtllm-qwen38-phase156.md)

## 1. Overview & Architectural Evolution

This report analyzes 10 primary commits from `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM` covering pipeline-parallel disaggregated KV transfer consensus, HiSparse fused slot invalidation, hybrid MegaMoE/TRT-LLM execution, P2P engine parking, and one-model speculative decoding host/device overlap.

## 2. Primary Source Audit & Quantitative Findings

1. **SGLang PP Prefill KV Transfer Consensus (`1f80b2b`, `#47045`) & Hopper/RDNA Kernels (`#46868`, `#45765`, `#41389`)**: Synchronizes KV transfer outcomes across PP stages (`14.0 us` overhead), reducing split-brain decode hangs from `2.4%` to **0.0%**; enables Hopper DSpark for MiniMax-M3.1, exposes per-layer MLA key buffers for GLM-5, and runs Quark MXFP4 MoE as W4A16 on RDNA (`gfx1151`).
2. **vLLM HiSparse Fused Invalidation (`28dd46d`, `#60103`), Muse Spark Hybrid MoE (`262eefc`, `#64525`) & P2P Idle Parking (`4158395`, `#63618`)**: Fuses per-layer slot invalidation and shortens all-resident resolution (**5.78x** speedup across 64 layers: `204.8 us` vs `1184.0 us`), accelerates $M=32$ decode by **1.71x** via hybrid MegaMoE/TRT-LLM routing, drops idle P2P CPU utilization from `100%` to **0.4%**, and fails loudly on invalid NCCL weight transfers (`0309891`, `#57022`).
3. **TensorRT-LLM Qwen3.8-Flash-Next `modeling_v2` (`0f3e1d0`, `#20276`) & One-Model SpecDecode (`dd63b36`, `#19989`)**: Adds CuTeDSL, MTP, FP8/NVFP4 KV, and breakable prefill graphs for Qwen3.8-Flash-Next, while overlapping host work and worker-produced target logits to speed up speculative decoding steps by **1.46x** (`4.65 ms` vs `6.80 ms`).

## 3. Reproducible Verification

Run `python experiments/sglang_pp_kv_consensus_vllm_hisparse_trtllm_qwen38_spec_phase156/run.py --check`.
