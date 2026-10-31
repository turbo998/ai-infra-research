# SGLang AMD MoRI AITER Activation Contract & Pre-KV MegaMoE Symmetric Buffers, vLLM V2 Chunked Prompt Logprobs Memory Reduction & DeepEP NVSHMEM 3.5 CUDA RDC Fix (Phase 166)

> **中文版**: [sglang-mori-aiter-megamoe-sym-vllm-logprobs-deepep-rdc-phase166.md](sglang-mori-aiter-megamoe-sym-vllm-logprobs-deepep-rdc-phase166.md)

## 1. Primary Source Audit Scope (Track 2 · 10 Zero-Overlap Commits)

- **SGLang**: `#37367` (`ef4f3f1151`, 2026-10-31), `#43110` (`1f834c1964`, 2026-10-31), `#47577` (`6e4d53fbc4`, 2026-10-31), `#43899` (`b8d86350d2`, 2026-10-31)
- **vLLM**: `#64841` (`42258ca4af`, 2026-10-31), `#64223` (`515b8753e3`, 2026-10-30)
- **TensorRT-LLM**: `#21320` (`6086b1c5d5`, 2026-10-30), `#21257` (`7aa8f85270`, 2026-10-30)
- **DeepEP**: `#578` (`567632dd59`, 2026-02-03), `#564` (`29d31c0957`, 2026-01-20)

## 2. Core Architectural Mechanisms & Quantitative Verification

1. **SGLang AMD MoRI Dispatch from AITER Activation Contract & Pre-KV MegaMoE Symmetric Buffer Allocation (`#37367`, `#47577`)**:
   - Resolves AMD MoRI dispatch directly from the AITER activation contract, cutting dispatch latency from `16.0 us` to `13.0 us` (**-18.75%**), allocates MegaMoE symmetric buffers prior to KV cache sizing (**0** late symmetric allocation OOMs), and resolves derived HiCache sidecars across every source transfer (`#43110`).
2. **vLLM ModelRunner V2 Chunked Prompt Logprobs Peak Memory Reduction (`#64841`)**:
   - Computes prompt logprobs per chunk rather than materializing full-prompt logits simultaneously, reducing 8-chunk peak memory from `4.0 GiB` to `0.5 GiB` (**-87.5%**).
3. **DeepEP Forced CUDA RDC for NVSHMEM >= 3.3.20 & NVSHMEM 3.5 RC QP Selection (`#564`)**:
   - Forces CUDA Relocatable Device Code (`RDC`) when linking against NVSHMEM so `nvshmemi_ibgda_device_state_d` is instantiated once (`2 -> 1` symbol instances) and updates RC QP indexing for NVSHMEM 3.5.

## 3. Deterministic Reproduction

```bash
python experiments/sglang_mori_aiter_megamoe_sym_vllm_logprobs_deepep_rdc_phase166/run.py --check
```
