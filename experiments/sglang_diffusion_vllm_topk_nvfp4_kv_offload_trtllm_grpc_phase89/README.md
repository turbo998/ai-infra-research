# Phase 89 Track B: SGLang Diffusion & GLM Parser, vLLM Persistent TopK & NVFP4/ROCm MoE, and TensorRT-LLM UPB gRPC

Deterministic Python 3.12 standard-library model reproducing the architectural invariants across the 10 Phase 89 Track B inference engine commits (`sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`).

## Usage

```bash
python experiments/sglang_diffusion_vllm_topk_nvfp4_kv_offload_trtllm_grpc_phase89/run.py
python experiments/sglang_diffusion_vllm_topk_nvfp4_kv_offload_trtllm_grpc_phase89/run.py --check
```
