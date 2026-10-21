# TensorRT-LLM FLUX.1-dev NVFP4 + CUTEDSL FP8 MHA, Triton Custom-Mask FP8, Gemma4 Heterogeneous KV, vLLM XPU OSA & SGLang TP KV Release

Deterministic Python 3.12 standard-library reproduction for Phase 47 Topic 2 (`q-phase47-trtllm-flux-nvfp4-cutedsl-fp8-custom-mask-gemma4-vllm-xpu-sglang-tp-release`).

## Covered Upstream Commits
- `NVIDIA/TensorRT-LLM@5d6fbb2` (`#18396`), `5f90e89` (`#20387`), `e8127c7` (`#18547`)
- `vllm-project/vllm@35d0166` (`#62244`), `550c1cb` (`#59703`)
- `sgl-project/sglang@b57b5e2` (`#45453`), `5e79333` (`#45439`)

## Run
```bash
python experiments/trtllm_flux_nvfp4_cutedsl_fp8_custom_mask_gemma4_vllm_sglang/run.py --check
```
