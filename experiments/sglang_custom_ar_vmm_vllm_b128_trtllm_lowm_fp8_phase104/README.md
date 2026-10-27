# SGLang MNNVL `custom_all_reduce_v2` VMM Gate Removal & DSA Squeeze, vLLM ROCm W4A16 `b128` LDS Loads & Qwen3.8 QSA, and TRT-LLM Low-M FP8 Decode GEMM (Phase 104 · Priority 113)

Deterministic Python 3.12 stdlib experiment verifying:
- `sgl-project/sglang@de26bb16a9`, `8db96fd402`, `449617bc94` & `9d45fdda46`: MNNVL `custom_all_reduce_v2` `uses_vmm` gate removal (`48.0 -> 14.0 us`, `3.4286x` 16-GPU speedup) & DSA `.squeeze(0)` token-first output preservation (`0` DP padding shape errors).
- `vllm-project/vllm@83909ad84b`, `ca8dbccf20`, `a91a1d06b7` & `e7fa26d5c8`: ROCm W4A16 skinny GEMM `b128` LDS activation loads (`-50.0%` LDS instructions `64 -> 32`, `16 B` alignment), Triton launch-grid forwarding, and Qwen3.8-Flash-Next BF16 QSA indexer preservation.
- `NVIDIA/TensorRT-LLM@0712980587` & `a1eebc47dd`: Blackwell SM100/103/107 `fp8_block_scale_low_m_gemm` decode projection tactic tables (`87.5% -> 50.0%` M-padding waste at `M=16`, `2.1333x` speedup `12.8 -> 6.0 us`) and `Responses` `namespace::tool` replay restoration (`100%` retention).

```powershell
python experiments\sglang_custom_ar_vmm_vllm_b128_trtllm_lowm_fp8_phase104\run.py --check
```
