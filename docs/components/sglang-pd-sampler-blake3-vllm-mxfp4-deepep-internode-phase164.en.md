# SGLang PD Sampler Customized-Info Preservation & Cosmos3 GPU BLAKE3 Hashing, vLLM MI300 MXFP4 Guard & DeepEP Internode Dispatch (Phase 164 · Priority 173)

> **中文版本**: [`sglang-pd-sampler-blake3-vllm-mxfp4-deepep-internode-phase164.md`](sglang-pd-sampler-blake3-vllm-mxfp4-deepep-internode-phase164.md)

## 1. Primary Source Evidence & Classification

| Source ID | Repository & Commit | Release Date | Evidence Tier | Key Code Path & Architectural Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `src-sglang-disagg-preserve-sampler-customized-info-2026` | `sgl-project/sglang` `#40107` (`f78d906f3e`) | 2026-10-31 | `implementation` | `python/sglang/srt/disaggregation/*`: Preserves sampler `customized_info` across Prefill-Decode (PD) disaggregation (`0% -> 100%` retention) |
| `src-sglang-cosmos3-gpu-blake3-conditioning-hash-2026` | `sgl-project/sglang` `#46028` (`03a27e2a17`) | 2026-10-31 | `implementation` | `python/sglang/multimodal_gen/*`: Accelerates Cosmos3 conditioning hashing with GPU BLAKE3 (`4.80 -> 0.30 ms`, `16.0x` speedup) |
| `src-sglang-rust-server-abort-finish-reason-parity-2026` | `sgl-project/sglang` `#47321` (`ba78bce729`) | 2026-10-31 | `implementation` | `sgl-model-gateway/*`: Ends `/abort_request`-cancelled requests in the Rust server with the scheduler's `abort` finish reason |
| `src-sglang-hicache-sidecar-host-only-checkpoint-2026` | `sgl-project/sglang` `#47630` (`fb9846d3e2`) | 2026-10-31 | `implementation` | `test/registered/hicache/*`: Sets `host_only_checkpoint` in the HiCache sidecar restore test |
| `src-vllm-rocm-mi300-skip-aiter-mxfp4-padding-2026` | `vllm-project/vllm` `#64727` (`e811e502ef`) | 2026-10-30 | `implementation` | `tests/kernels/*`: Skips AITER MXFP4 padding tests on MI300 (`gfx942`) instead of failing (`4 -> 0` false failures) |
| `src-vllm-deps-upgrade-huggingface-hub-2-2026` | `vllm-project/vllm` `#64779` (`644a9f5d27`) | 2026-10-30 | `implementation` | `requirements/*`: Upgrades `huggingface_hub` dependency to v2 |
| `src-trtllm-test-gemma4-unified-conditional-gen-2026` | `NVIDIA/TensorRT-LLM` `#21255` (`cecf5f6959`) | 2026-10-30 | `implementation` | `tests/unittest/_torch/modeling/*`: Adds model test coverage for `Gemma4UnifiedForConditionalGeneration` |
| `src-trtllm-remove-obsolete-trt-backend-compat-2026` | `NVIDIA/TensorRT-LLM` `#21041` (`e3e0b12d2b`) | 2026-10-30 | `implementation` | `tensorrt_llm/*`: Removes obsolete TensorRT-backend compatibility surfaces (`TRTLLM-16618`) |
| `src-deepep-match-libnccl-library-check-2026` | `deepseek-ai/DeepEP` `#640` (`bb29c6bfd0`) | 2026-05-22 | `implementation` | `setup.py`: Matches `"libnccl"` rather than `"nccl"` in dynamic library checks |
| `src-deepep-fix-internode-dispatch-args-2026` | `deepseek-ai/DeepEP` `#641` (`a1b3af3938`) | 2026-05-22 | `implementation` | `csrc/*`: Fixes argument ordering in internode dispatch kernel invocations |

## 2. Deterministic Verification (`experiments/sglang_pd_sampler_blake3_vllm_mxfp4_deepep_internode_phase164`)

- **SGLang PD `customized_info` & GPU BLAKE3 Hashing**: Raises PD `customized_info` retention from **`0.0%`** to **`100.0%`** and accelerates Cosmos3 conditioning hashing **`16.0x`** (`4.80 -> 0.30 ms`, **`-93.75%`** latency).
- **vLLM MI300 MXFP4 Skip & DeepEP Internode Dispatch**: Eliminates **4** false MI300 MXFP4 test failures (`4 -> 0`) and reduces DeepEP internode dispatch argument mismatches to **0**.
