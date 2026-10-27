# Phase 111 Track B Experiment: SGLang DFlash MLA & Async NaN Probe, vLLM RDNA/MORI-IO/Overlaid KV, and TRT-LLM Helix CP Consensus & MiniMax-H3 Fused RoPE

Deterministic Python standard-library model covering:
1. **SGLang (`b068417bfcc1`, `ee5443f53026`, `302d5e7b8ca9`, `c43abd6d9e38`)**:
   - DFlash serving SpecForge MLA drafts through target MLA attention with `set_mla_kv_buffer_prefix_valid` slot-0 redirection for uncommitted tokens.
   - Single-kernel Triton asynchronous NaN validation (`_assert_no_nan_kernel`), NUMA physical CPU budget caching, and DiT breakable CUDA graph (`bcg`) caller-allocated output buffers.
2. **vLLM (`9af236a33281`, `94f56ce92cc4`, `3d81a510f9eb`, `8701c9b7513d`)**:
   - ROCm RDNA `kernel_paged_attention_2d` warp/stage tuning (`num_stages=1`, `num_warps=8/16`), SM90 GDN decode `BV=4` for FP32 state, MORI IO stride-aware `block_len`, and geometry-keyed `copy_kv_cache_blocks_inplace` for overlaid Mamba+Attention KV caches.
3. **TensorRT-LLM (`94348c4de6eb`, `af32051e1926`)**:
   - `align_gen_admission_across_cp` gen-side KV transfer consensus across Helix CP ranks under attention-DP, and MiniMax-H3 BF16 fused per-head QK-norm + partial split-half RoPE.

## Run
```bash
python experiments/sglang_dflash_mla_vllm_moriio_trtllm_helix_cp_phase111/run.py --check
```
