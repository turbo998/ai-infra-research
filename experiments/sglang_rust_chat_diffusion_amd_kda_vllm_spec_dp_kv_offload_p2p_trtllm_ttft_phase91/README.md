# Phase 91 Track B: SGLang Rust Chat & Diffusion Disaggregation, vLLM Spec Decode DP & P2P KV Offload, TensorRT-LLM KV Scan & TTFT-Percentile GEN Admission

Deterministic Python 3.12 standard-library model covering 10 first-hand commits:

1. `sgl-project/sglang#45449` (`911693e3f8`): Unified chat rendering, tokenization, and serving in `rust-server` via `sglang-processor`, removing 6 duplicate Rust dependencies (`dynamo-parsers`, `dynamo-protocols`, `dynamo-renderer`, `dynamo-tokenizers`, `hf-hub`, `minijinja`).
2. `sgl-project/sglang#46060` (`fe4ccb37e4`): Atomic image input publication in `sglang.multimodal_gen` via temporary sibling file + `os.replace`.
3. `sgl-project/sglang#46193` (`d3e84c32e5`): Preserving GLM pixel payloads for the head worker across disaggregated diffusion dynamic batch splits.
4. `sgl-project/sglang#45710` (`c8398cb3a3`): Native HIP chunked-prefill kernel (`kda_prefill_gfx950.cu`) for Kimi Delta Attention on AMD `gfx950` (MI350X / MI355X).
5. `vllm-project/vllm#62301` (`29be2f2560`): Preserving `parallel_config.data_parallel_index` on standalone speculative-decoding draft models.
6. `vllm-project/vllm#60574` (`a64fc94081`): Preserving newly queued KV offload reload jobs when a prefix-cache reset preempts and reloads a request in the same scheduler step.
7. `vllm-project/vllm#63664` (`f682f1dc32`): Routing P2P `ConnectAckMsg` through `_do_send` so send failures immediately mark the peer session dead.
8. `vllm-project/vllm#63498` (`71294c2e9a`): Setting UCX `engine_config="TLS=^rc_gda"` on the P2P NIXL agent so host-DRAM tiering excludes GPU-direct `rc_gda` lanes.
9. `NVIDIA/TensorRT-LLM#20928` (`c3c1ad3ab8`): Accelerating post-warmup KV cache NaN/Inf scan with `_INVALID_VALUE_SCAN_ELEMS = 1 << 26` (`67,108,864` elements per call, `16x` fewer slices than 256-page chunks) and caching unsupported `isinf` checks on FP8.
10. `NVIDIA/TensorRT-LLM#20736` (`dd75334918`): Disaggregated GEN admission queue ordered by orchestrator arrival (`orig_elapsed_s`, `TLLM_GEN_ADMISSION_TTFT_PERCENTILE`) and exact context-chunk replay under SWA scratch reuse after a skipped forward.
