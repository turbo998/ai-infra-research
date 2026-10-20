# SGLang DeepSeek-V4 DSpark TF32 mHC Row Threshold, FlashInfer GDN TMEM Load Fencing, and TensorRT-LLM MegaMoE CuteDSL TEP Contract

Deterministic Python 3.12 standard-library model verifying:
1. **SGLang DeepSeek-V4 DSpark Draft-Stage TF32 mHC Path (`sgl-project/sglang` `c2e57424`)**: Lowering `_mix_stats_impl` threshold on SM100 from `rows >= 128` to `rows >= 32` increases draft batch fast-path hits from `2/7` to `6/7` (`1.6373x` speedup) and refreshes mHC norm weight caches via `post_load_weights()`.
2. **FlashInfer Blackwell Gated Delta Net TMEM Load Fencing (`flashinfer-ai/flashinfer` `80a44189`)**: Inserting `cute.arch.fence_view_async_tmem_load()` before `ks_handle.release()` and `nv_handle.release()` eliminates `16/16` asynchronous TMEM read-before-overwrite hazards across 8 chunks.
3. **TensorRT-LLM MegaMoE CuteDSL TEP Global Routed Output Contract (`NVIDIA/TensorRT-LLM` `46d9966b`)**: Coordinating `accepts_global_routed_output` and `returns_global_routed_output` eliminates `4x` (`tp_size=4`) routed output inflation under TEP (`0.0` error across both opt-in and generic model wrappers).
