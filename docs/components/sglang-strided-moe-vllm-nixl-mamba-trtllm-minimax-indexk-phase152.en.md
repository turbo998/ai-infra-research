# SGLang Strided-Grid MoE Padded Row Fill, vLLM NIXL Mamba Block-Outermost Stride & TensorRT-LLM MiniMax-M3 Sparse Index-K Cache (Phase 152)

> **Evidence Classification**: Primary source implementation audit (`implementation`) + deterministic stdlib offline simulation (`own-offline`).

## 1. 10 Primary Commit Audit Matrix (Zero-Overlap)

| # | Repository | Commit (`SHA`) | Date | Core Engineering Breakthrough & Code Path |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `sgl-project/sglang` | [`2830808`](https://github.com/sgl-project/sglang/commit/2830808b46822216df7cc329fa44bc1caaa67be2) | 2026-10-30 | `[Refactor] scheduler: Write every request finish state through Req methods` (`#43896`). |
| 2 | `sgl-project/sglang` | [`add6f6d`](https://github.com/sgl-project/sglang/commit/add6f6d2c3cb5a94fd0f64a025614c1710722d97) | 2026-10-29 | `[Refactor] detokenizer: Move the incremental decode state and step into incremental_detokenize.py` (`#47338`). |
| 3 | `sgl-project/sglang` | [`20036b7`](https://github.com/sgl-project/sglang/commit/20036b7006ca5433b6e35c4e244684829c317873) | 2026-10-29 | `[Perf] moe: Fill padded rows of large narrow buffers with a strided grid` (`#47115`). |
| 4 | `sgl-project/sglang` | [`c805455`](https://github.com/sgl-project/sglang/commit/c805455f71be19175db0c849f834e843366d4a79) | 2026-10-30 | `[Fix] minimax_h3: Support out-of-tree platforms` (`#44725`). |
| 5 | `vllm-project/vllm` | [`8084f06`](https://github.com/vllm-project/vllm/commit/8084f0618162eb1b4c80448a5b593c8fc3d718bf) | 2026-10-29 | `[Bugfix][KV Connector][NIXL] Fix Mamba region stride on block-outermost layouts` (`#64582`). |
| 6 | `vllm-project/vllm` | [`ed14230`](https://github.com/vllm-project/vllm/commit/ed142309b58359333e34c3a44a2ee3ee773e9185) | 2026-10-30 | `[Bugfix] Skip more weight formats in the tokenizer/processor snapshot` (`#63922`). |
| 7 | `vllm-project/vllm` | [`c05e9ed`](https://github.com/vllm-project/vllm/commit/c05e9edc0134832b453d60b6fa78823c94879f1c) | 2026-10-30 | `[Bugfix][Benchmark] Count tool call deltas as generated tokens in openai-responses` (`#62753`). |
| 8 | `vllm-project/vllm` | [`dc42019`](https://github.com/vllm-project/vllm/commit/dc42019e7c33d7203be8b77ef435e185cb73c575) | 2026-10-30 | `[Bugfix][Frontend] Leave GLM-4.7 array and object tool arguments unpinned` (`#64615`). |
| 9 | `NVIDIA/TensorRT-LLM` | [`c492171`](https://github.com/NVIDIA/TensorRT-LLM/commit/c49217126d27a72fc4be59dc615292d19ca0fd06) | 2026-10-29 | `MiniMax-M3 VL: forward video metadata so videos get timestamps` (`#20695`). |
| 10 | `NVIDIA/TensorRT-LLM` | [`f7e0b91`](https://github.com/NVIDIA/TensorRT-LLM/commit/f7e0b912b5a2ef40fd8c298fc8e2fb569bc4db4d) | 2026-10-29 | `Derive MiniMax-M3 index-K cache layers from the model config` (`#20693`). |

## 2. Deterministic Offline Verification (`own-offline`)

Run `python experiments/sglang_strided_moe_vllm_nixl_mamba_trtllm_minimax_phase152/run.py --check`:
- **SGLang Strided-Grid MoE Fill Speedup**: **`1.9676x`** (`36.4 -> 18.5 us`)
- **vLLM NIXL Mamba Block-Outermost Stride Fix**: Eliminates `511/512` corrupted Mamba state blocks (`32 KB -> 768 KB` block stride)
- **vLLM Tokenizer Snapshot Bandwidth Saving**: **`99.86%`** (`28.5 -> 0.04 GB`)
- **TensorRT-LLM MiniMax-M3 Sparse Index-K Cache Saving**: **`75.0%`** (`20` vs `80` layers)
