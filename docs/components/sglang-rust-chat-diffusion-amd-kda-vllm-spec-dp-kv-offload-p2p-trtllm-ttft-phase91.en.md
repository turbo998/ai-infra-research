# Phase 91 Track B: SGLang Rust Chat Processor & Atomic Diffusion Image Publish / GLM Pixel Preservation, AMD gfx950 HIP KDA Prefill, vLLM Spec Decode DP & P2P KV Offload Hardening, and TensorRT-LLM 64M KV Scan & TTFT-Percentile GEN Admission

> **中文版本**: [sglang-rust-chat-diffusion-amd-kda-vllm-spec-dp-kv-offload-p2p-trtllm-ttft-phase91.md](sglang-rust-chat-diffusion-amd-kda-vllm-spec-dp-kv-offload-p2p-trtllm-ttft-phase91.md)  
> **Phase Status**: Phase 91 (`priority: 100 — 100-Priority Milestone!`) · Question `q-sglang-rust-chat-diffusion-amd-kda-vllm-spec-dp-kv-offload-p2p-trtllm-ttft-phase91` · Claim `claim-sglang-rust-chat-diffusion-amd-kda-vllm-spec-dp-kv-offload-p2p-trtllm-ttft-phase91`  
> **Reproducible Experiment**: [`experiments/sglang_rust_chat_diffusion_amd_kda_vllm_spec_dp_kv_offload_p2p_trtllm_ttft_phase91/`](../../experiments/sglang_rust_chat_diffusion_amd_kda_vllm_spec_dp_kv_offload_p2p_trtllm_ttft_phase91/)

---

## 1. First-Hand Source Evidence Matrix (10 Upstream Commits)

| # | Repository & Commit | Date | Evidence Class | Core Architectural Mechanism |
|---|---|---|---|---|
| 1 | `sgl-project/sglang` [`911693e3f8`](https://github.com/sgl-project/sglang/commit/911693e3f8cc5d8dbdd5490934f448cb7db9722f) (`#45449`) | 2026-10-26 | `implementation` | `rust-server` renders, tokenizes, and serves chat via `sglang-processor`, dropping 6 duplicate Rust crates (`dynamo-parsers`, `dynamo-protocols`, `dynamo-renderer`, `dynamo-tokenizers`, `hf-hub`, `minijinja`) and forwarding `default_chat_template_kwargs` |
| 2 | `sgl-project/sglang` [`fe4ccb37e4`](https://github.com/sgl-project/sglang/commit/fe4ccb37e400ef857816a18f383f965661139ff4) (`#46060`) | 2026-10-26 | `implementation` | `sglang.multimodal_gen` publishes image inputs atomically via sibling temp file + `os.replace` (`save_image_bytes_to_path`) |
| 3 | `sgl-project/sglang` [`d3e84c32e5`](https://github.com/sgl-project/sglang/commit/d3e84c32e5858c553463686f5f9471d68702efbe) (`#46193`) | 2026-10-26 | `implementation` | Disaggregated diffusion preserves GLM pixel payloads for the head worker across dynamic batch splits |
| 4 | `sgl-project/sglang` [`c8398cb3a3`](https://github.com/sgl-project/sglang/commit/c8398cb3a378d62451023482ca3d20713aa59b78) (`#45710`) | 2026-10-26 | `implementation` | Native HIP chunked-prefill kernel (`kda_prefill_gfx950.cu`, `+2898` lines) for Kimi Delta Attention on AMD `gfx950` (MI350X / MI355X) |
| 5 | `vllm-project/vllm` [`29be2f2560`](https://github.com/vllm-project/vllm/commit/29be2f2560f0a4bb86796340928f8b2d56ab2d98) (`#62301`) | 2026-10-26 | `implementation` | Standalone speculative-decoding draft models preserve `parallel_config.data_parallel_index` from the target model |
| 6 | `vllm-project/vllm` [`a64fc94081`](https://github.com/vllm-project/vllm/commit/a64fc94081494957dbdc0c3605b4b1ab2867bc0e) (`#60574`) | 2026-10-26 | `implementation` | KV Offload scheduler preserves newly queued reload jobs when a prefix-cache reset preempts and resumes a request in the same step |
| 7 | `vllm-project/vllm` [`f682f1dc32`](https://github.com/vllm-project/vllm/commit/f682f1dc322bbbc69c390209e15fb90910f9d4c0) (`#63664`) | 2026-10-26 | `implementation` | P2P KV tiering session routes `ConnectAckMsg` through `self._do_send(...)` so send failures immediately mark the session dead |
| 8 | `vllm-project/vllm` [`71294c2e9a`](https://github.com/vllm-project/vllm/commit/71294c2e9ac41ed710b6eb4ca41cba3793705345) (`#63498`) | 2026-10-26 | `implementation` | P2P NIXL transport passes `engine_config="TLS=^rc_gda"` when creating the UCX backend to exclude GPU-direct `rc_gda` lanes from host-only P2P tiering |
| 9 | `NVIDIA/TensorRT-LLM` [`c3c1ad3ab8`](https://github.com/NVIDIA/TensorRT-LLM/commit/c3c1ad3ab84c811a5c2841a946c841dd3c8c2b1c) (`#20928`) | 2026-10-26 | `implementation` | Post-warmup KV cache NaN/Inf scan batches `_INVALID_VALUE_SCAN_ELEMS = 1 << 26` (`67,108,864` elements per call, `16x` fewer launches) and caches unsupported `isinf` checks on FP8 |
| 10 | `NVIDIA/TensorRT-LLM` [`dd75334918`](https://github.com/NVIDIA/TensorRT-LLM/commit/dd75334918500a9c0ad662b156afc9b49aeb234b) (`#20736`) | 2026-10-26 | `implementation` | Disaggregated serving adds `GenAdmissionWaitingQueue` ordered by orchestrator arrival (`orig_elapsed_s`, `TLLM_GEN_ADMISSION_TTFT_PERCENTILE`) and enforces exact pregrown context-chunk replay under SWA scratch reuse |

---

## 2. Implementation Deep Dive

### 2.1 vLLM P2P KV Tiering & Speculative Decode DP Invariants (`#62301`, `#60574`, `#63664`, `#63498`)
- **Speculative Decode DP Alignment (`#62301`)**: Standalone draft models previously constructed a fresh `ParallelConfig` that defaulted `data_parallel_index = 0` on non-zero DP replicas; copying `self.vllm_config.parallel_config.data_parallel_index` aligns draft KV cache and communication groups with the target replica.
- **P2P NIXL `TLS=^rc_gda` (`#63498`)**: Because P2P KV tiering operates on host `memoryview` buffers, allowing UCX to negotiate `rc_gda` (GPU-Direct Async) lanes causes stalled peers to fail GPU lane establishment. Passing `engine_config="TLS=^rc_gda"` restricts UCX to host transports.

### 2.2 TensorRT-LLM 64M-Element KV Scan & TTFT-Percentile GEN Admission (`#20928`, `#20736`)
- **64M-Element NaN/Inf Scan (`#20928`)**: Scanning KV caches in `1 << 26`-element flattened chunks cuts kernel launch overhead by `16x` compared to 256-page slices while avoiding repeated `isinf` exceptions on FP8 tensors.
- **TTFT-Percentile GEN Admission & SWA Replay (`#20736`)**: Ordering the GEN waiting queue by orchestrator arrival (`orig_elapsed_s`) prevents long-context prefill requests from being starved behind short requests that arrived later at the orchestrator but finished prefill earlier. When SWA scratch page reuse is active and a forward step is skipped after growing KV capacity, `_pregrown_context_tokens` replays the exact pregrown chunk size so sliding-window page indices remain valid.
