# SGLang DeepSeek-V4.1 In-Place NVLink Engram Reads, AMD MoRI DSpark PD & `gfx950` Batch-Invariant BF16 GEMM, Integer Custom All-Reduce Guard & PyNccl Check, plus vLLM Mamba SSU/Helion Config Packaging and TRT-LLM OpenEngine gRPC Video Inputs (Phase 131 · Priority 140)

> **中文版**: [sglang-engram-nvlink-mori-gfx950-vllm-helion-trtllm-video-phase131.md](sglang-engram-nvlink-mori-gfx950-vllm-helion-trtllm-video-phase131.md)  
> **Evidence Anchor**: `ev-sglang-engram-nvlink-mori-gfx950-vllm-helion-trtllm-video-phase131-2026` (`own-offline` + `implementation`)  
> **Reproducible Experiment**: `python experiments/sglang_engram_nvlink_mori_gfx950_vllm_helion_trtllm_video_phase131/run.py --check`

---

## 1. Primary Source Matrix

- **`sgl-project/sglang` (`54d2f65`, `b85d6a8`, `1e7e192`, `c240169` & `990ee89`)**: Reading DeepSeek-V4.1 GPU Engram tables in place over NVLink eliminates **`58,720,256 B` (`56.0 MiB`)** of intermediate staging copies (**`142.0 -> 86.5 us`, `1.6416x` speedup**); enabling AMD MoRI for DeepSeek-V4.1 DSpark PD disaggregation; tuning AMD `gfx950` batch-invariant BF16 matmul (**`215.0 -> 154.0 us`, `1.3961x`**); keeping integer (`int32`/`int64`) tensors off floating-point Custom All-Reduce kernels (**`1 -> 0` metadata corruption**); and guarding unavailable `PyNccl` enablement.
- **`vllm-project/vllm` (`df028df`, `41dd82e` & `deb1a97`) & `NVIDIA/TensorRT-LLM` (`7b623fb` & `b41f992`)**: Shipping Mamba SSU and Helion tuned kernel configs in vLLM wheel `package_data` avoids un-tuned fallback tiles (**`118.0 -> 76.0 us`, `1.5526x`**), fixing XPU `TP=1` FP8 `rms_quant_fusion`, and restoring `--input-len/--output-len/--prefix-len` CLI flags; TRT-LLM OpenEngine gRPC native binary video inputs compress wire payload **`10.7025x`** vs JSON Base64 (`265.42 MB -> 24.80 MB`) alongside automatic GPU-count `TP/EP` selection for DeepSeek-V4 Flash/Pro/DSpark.
