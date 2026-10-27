# Phase 103 Track B: SGLang DSpark / HiCache / DeepEP v2, vLLM ROCm Sparse MLA, TensorRT-LLM & AI Infra Innovators

Deterministic Python 3.12 standard-library model reproducing:
- `sgl-project/sglang@a09c1a096c` (#46309): DeepEP v2 `_masked_token_capacity` shrinking CUDA Graph masked GEMM buffer by up to `32.0x` (`1,792.0 -> 56.0 MiB` at `local_tokens=4`).
- `sgl-project/sglang@2a7987f900` (#44081) & `24d55621a3` (#46404): PD HiCache load-back early-send hazard elimination & DSpark simulated-acceptance bonus row selection.
- `vllm-project/vllm@939ff73503` (#63785), `8d88bc5d81` (#60154), `f8f7634cb4` (#63342) & `NVIDIA/TensorRT-LLM@8bd66f6a71` (#20904), `105401ea8d` (#20467).
- Quantitative architectural modeling of **Fireworks AI**, **Together AI**, **Cerebras WSE-3**, and **Groq LPU** vs **NVIDIA Blackwell B200/GB300**.

## Usage
```bash
python experiments/sglang_dspark_hicache_vllm_trtllm_phase103/run.py
python experiments/sglang_dspark_hicache_vllm_trtllm_phase103/run.py --check
```
