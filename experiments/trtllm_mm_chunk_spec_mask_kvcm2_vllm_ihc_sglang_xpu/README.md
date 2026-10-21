# TensorRT-LLM Chunk-Scoped Multimodal Scheduling, Blackwell `SwapsMmaAb` Speculative Mask Sizing, KVCM2 Progressive Host Commit, vLLM `libgomp` Fork & ROCm HY-V4 iHC, and SGLang Intel XPU Disaggregated KV

Deterministic Python 3.12 standard-library verification for **Phase 46 (`priority: 55`)** Topic 2 (`q-phase46-trtllm-mm-chunk-spec-mask-kvcm2-vllm-ihc-sglang-xpu`).

## First-Hand Upstream Commits
- `TensorRT-LLM@b47eb8fe6b0965a85adf01121f9f0f2a1e79c7c2` (`#19884`): Chunk-scoped multimodal encoder budgeting (`ContextChunk(start, end)`) after KV prefix reuse resolution and `TensorLRUCache.releasable_bytes`.
- `TensorRT-LLM@85b1b8323b78a4717aeb772aa66eed9b8346d687` (`#20223`): Blackwell speculative decoding tree custom-mask upper bound across `KeepsMmaAb` and `SwapsMmaAb` (`4.0x` buffer capacity over legacy `KeepsMmaAb`-only sizing).
- `TensorRT-LLM@0cd9d0b5415b82696ecbaba8b0b6034aa9c09997` (`#19650`): KVCM2 progressive host tier commit (`MAP_NORESERVE` upfront virtual reservation + `2 MiB -> 32 MiB` background commit units, `2048x` smaller synchronous startup commit at `64 GiB`).
- `vllm@40210d8317bbad8d05124b260e75677daf33e091` (`#62884`) & `vllm@833349105b9581d19487295454c19c924e8a3b37` (`#58509`): `omp_pause_resource_all(omp_pause_hard=2)` before `os.fork()` and ROCm HY-V4 `triton_ihc_post_pre_rms_norm` cross-layer fusion (`1.6x` HBM traffic reduction).
- `sglang@8e3e2980db83a1b020a6d273a2b653891af81bfc` (`#40861`), `sglang@f8fd41c36e493ec762e60f447c50c88fb8af3ec1` (`#43363`), & `sglang@2c9d9190713285552d7ea2bb264664c90639c824` (`#45437`): Intel XPU `ptr & (1 << 63)` kernel-space pointer detection and host staging for NIXL/Mooncake KV transfer, FlashInfer CUTLASS MoE `tune_max_num_tokens = min(prefill_tokens * attn_dp_size, 8192)`, and `SGLANG_ZMQ_SOURCE_IP`.

## Usage
```bash
python experiments/trtllm_mm_chunk_spec_mask_kvcm2_vllm_ihc_sglang_xpu/run.py --check
```
