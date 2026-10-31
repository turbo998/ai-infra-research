# SGLang Blackwell Subtile Diffusion Attention, vLLM Zero-Clone Weight Loader, TensorRT-LLM Single-GPU GDN Alignment & PyTorch AOTI XPU Deep Dive (Phase 159)

> **Evidence Mode**: `implementation` (fixed upstream commit source audit) + `own-offline` (deterministic Python 3.12 standard-library model in `experiments/sglang_diffusion_subtile_vllm_weight_clone_trtllm_gdn_phase159/run.py`). No GPU hardware benchmark executed.

## 1. Primary Source Matrix (10 Zero-Overlap Upstream Commits)

| Source ID | Repository | Commit (`2026-10`) | Key Technical Path & Implementation Change |
|---|---|---|---|
| `sglang-diffusion-blackwell-subtile-attention-select-2026` | `sgl-project/sglang` | `504f9cf720` (`#47301`) | Selects Blackwell subtile attention kernel based on execution mode and quality tier. |
| `sglang-rocm-diffusion-mp4-encode-vae-overlap-2026` | `sgl-project/sglang` | `0f92ddaefb` (`#47568`) | Overlaps MP4 video encoding with VAE decoding on ROCm diffusion pipelines. |
| `sglang-diffusion-minimax-h3-short-edge-warmup-2026` | `sgl-project/sglang` | `25c3fd8308` (`#47583`) | Warms up MiniMax-H3 at the short-edge resolutions specified by `--warmup-resolutions`. |
| `vllm-skip-redundant-weight-tensor-clones-2026` | `vllm-project/vllm` | `50f622bdcf` (`#63174`) | Skips redundant `.clone()` calls for weight loaders that already yield owned tensors. |
| `vllm-weights-mapper-transform-weights-2026` | `vllm-project/vllm` | `87cb835db2` (`#64869`) | Extends `WeightsMapper` to transform weight tensors during mapping. |
| `vllm-compile-mm-encoder-radio-navit-2026` | `vllm-project/vllm` | `10da92ff28` (`#63415`) | Fixes `compile_mm_encoder` support for RADIO and NaViT vision encoders. |
| `trtllm-hard-link-build-outputs-python-pkg-2026` | `NVIDIA/TensorRT-LLM` | `3d10fbf8ff` (`#21292`) | Hard-links build outputs into the Python package tree instead of duplicating binaries. |
| `trtllm-single-gpu-gdn-decode-alignment-2026` | `NVIDIA/TensorRT-LLM` | `8ba3e60cec` (`#21050`) | Replaces 8-GPU Ray pre-merge stage with single-GPU GDN decode alignment test. |
| `pytorch-aoti-xpu-weight-use-caching-allocator-2026` | `pytorch/pytorch` | `30e1e6e5a4` (`#201813`) | Supports `weight_use_caching_allocator` on XPU in AOTInductor. |
| `pytorch-inductor-symbolic-slice-bounds-size-oblivious-2026` | `pytorch/pytorch` | `6901ca42a3` (`#201763`) | Keeps slice bounds symbolic in backed size-oblivious mode in TorchInductor. |

## 2. Quantitative Analytical Summary (`own-offline`)
- **SGLang Diffusion Subtile & ROCm Overlap**: Achieves **$1.4178\times$** 1080p attention speedup ($43.1 \to 30.4\text{ ms}$) and **$33.09\%$** ROCm VAE+MP4 pipeline latency reduction ($6.80 \to 4.55\text{ s}$).
- **vLLM Zero-Clone Loader & MM Compile**: Reduces 405B weight loading peak RAM by **$48.89\%$** ($810.0 \to 414.0\text{ GB}$) and speeds up RADIO/NaViT encoder by **$1.6209\times$**.
- **TensorRT-LLM & PyTorch AOTI**: Cuts package disk usage by **$49.48\%$** ($9.60 \to 4.85\text{ GB}$) and saves **$46.77\times$** pre-merge GPU-hours.
