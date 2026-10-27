# Deep Dive: SGLang MNNVL `custom_all_reduce_v2` VMM Gate Removal & DSA Squeeze, vLLM ROCm W4A16 `b128` LDS Loads & TRT-LLM Blackwell Low-M FP8 Decode GEMM (Phase 104 · Priority 113)

> **中文版本**: [sglang-custom-ar-vmm-vllm-b128-trtllm-lowm-fp8-innovators-phase104.md](sglang-custom-ar-vmm-vllm-b128-trtllm-lowm-fp8-innovators-phase104.md)
> **Experiment**: [`experiments/sglang_custom_ar_vmm_vllm_b128_trtllm_lowm_fp8_phase104/run.py`](../../experiments/sglang_custom_ar_vmm_vllm_b128_trtllm_lowm_fp8_phase104/run.py) · **Summary**: [`summary.json`](../../experiments/sglang_custom_ar_vmm_vllm_b128_trtllm_lowm_fp8_phase104/summary.json)
> **7-Card Visual Report**: [`docs/reports/ai-infra-phase104-benchmark-cards.html`](../reports/ai-infra-phase104-benchmark-cards.html)

---

## 1. Evidence Classification & Upstream Anchors

| Source ID | Repo & Commit | Path | Classification |
|---|---|---|---|
| `src-sglang-custom-ar-remove-multinode-vmm-gate-2026` | `sgl-project/sglang@de26bb16a972d93f354218a29dbe177bce5655a4` | `python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py` | `implementation` |
| `src-sglang-dsa-token-first-sparse-output-squeeze-2026` | `sgl-project/sglang@8db96fd40202c068137fc56f524b2a53ce9a1026` | `python/sglang/srt/layers/attention/dsa_backend.py` | `implementation` |
| `src-sglang-dsv4-recipe-null-tool-effort-2026` | `sgl-project/sglang@449617bc94a5d4ae23ad62d0da2f842e711ca983` | `rust/sglang-processor/src/render/models/deepseek_v4.rs` | `implementation` |
| `src-sglang-qwen3-coder-mixed-union-preserve-2026` | `sgl-project/sglang@9d45fdda46d5918acb38b16d05df0b0985b286ba` | `python/sglang/srt/function_call/qwen3_coder_detector.py` | `implementation` |
| `src-vllm-rocm-w4a16-skinny-gemm-lds-b128-2026` | `vllm-project/vllm@83909ad84b3fe6e64536591fc26f02347b65fd63` | `csrc/rocm/skinny_gemms_int4.cu` | `implementation` |
| `src-vllm-triton-override-forward-launch-grid-2026` | `vllm-project/vllm@ca8dbccf204abfcdf9dcb178f7e108c06b224b0f` | `vllm/triton_utils/dispatcher.py` | `implementation` |
| `src-vllm-qwen38-flash-next-quant-qkv-bf16-qsa-2026` | `vllm-project/vllm@a91a1d06b72a53d6ba05840164b27a61e3b03d4e` | `vllm/models/qwen4_exp/nvidia/qsa.py` | `implementation` |
| `src-vllm-spec-decode-mhc-tilelang-refactor-2026` | `vllm-project/vllm@e7fa26d5c8d94636a4d0525e1a2462cb661ea331` | `vllm/model_executor/kernels/mhc/tilelang.py` | `implementation` |
| `src-trtllm-blackwell-fp8-block-scale-low-m-gemm-2026` | `NVIDIA/TensorRT-LLM@071298058728f652a416023cf1c94f2fc89368af` | `tensorrt_llm/_torch/kernels/blackwell/fp8_block_scale_low_m_gemm.py` | `implementation` |
| `src-trtllm-responses-namespace-tool-replay-2026` | `NVIDIA/TensorRT-LLM@a1eebc47dde5b8c1e1b4be503f889617d76ba040` | `tensorrt_llm/serve/responses_utils.py` | `implementation` |

---

## 2. Implementation Paths, Data Layout & Synchronization

### 2.1 SGLang MNNVL `custom_all_reduce_v2` VMM Gate Removal & DSA Squeeze (`de26bb16a9`, `8db96fd402`)
- **MNNVL Without `uses_vmm` (`de26bb16a9`)**: Multi-node MNNVL (`world_size = 16`) uses `torch.ops.symm_mem` workspace rather than single-node VMM graph-input mapping. Removing the `uses_vmm` gate restores MNNVL one-shot all-reduce (`48.0 -> 14.0 us`, **`3.4286x` speedup**).
- **DSA `.squeeze(0)` (`8db96fd402`)**: Squeezes `tilelang_sparse_fwd` output `(1, T, H, D) -> (T, H, D)` so DP-attention padding slices along the true token axis (`0` shape-mismatch crashes).

### 2.2 vLLM ROCm W4A16 `b128` LDS Loads & Qwen3.8-Flash-Next QSA (`83909ad84b`, `a91a1d06b7`)
- Widens ROCm W4A16 skinny GEMM LDS activation reads from `b64` (`8 B`) to `b128` (`16 B`) with `kActLdsAlign = 16 B`, cutting LDS load instructions per wave by **`50.0%` (`64 -> 32`)**, while keeping Qwen3.8-Flash-Next QSA indexer weights in `BF16` alongside quantized Q/K/V projections.

### 2.3 TensorRT-LLM Blackwell Low-M FP8 Decode GEMM & Responses Namespace Replay (`0712980587`, `a1eebc47dd`)
- Adds SM100/103 and SM107 tactic tables for `trtllm::fp8_block_scale_low_m_gemm`, reducing row-padding waste at $M=16$ from **`87.5%` ($M_{\text{tile}}=128$) to `50.0%` ($M_{\text{tile}}=32$)** and accelerating decode projections by **`2.1333x` (`12.8 -> 6.0 us`)**, while restoring stripped `namespace::tool` prefixes on Responses replay.
