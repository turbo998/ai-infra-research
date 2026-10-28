# Phase 118 Track B Experiment Suite

Deterministic Python stdlib simulation covering 10 upstream inference runtime commits (`priority: 127`):
- SGLang FlashAttention kernel dispatch (`586355a83b`, #46695)
- SGLang DeepSeek-V4.1 Engram & `environ.override` restoration (`655059819c`, #46896)
- SGLang diffusion/realtime post-init slot reservation (`9694a99d55`, #46252)
- SGLang diffusion cached named weight resolution (`5884e4fec6`, #46679)
- vLLM Spec Decode PCP batch-sharded draft logits guard (`820be160ce`, #64151)
- vLLM FlashInfer MLA DCP ragged spec-decode fix (`8697bbae4c`, #59392)
- vLLM KVCR adapter overhead reduction (`36f5ff3172`, #63889)
- vLLM ROCm Kimi-K3 fused MLA decode q-prep zero positions (`bb7ea479c6`, #63409)
- vLLM ROCm Qwen3.8-Flash-Next MI355X Quark MXFP4 recipe (`5314c58281`, #63332)
- TensorRT-LLM Kimi-K3 `helix_a2a` token-sliced MoE (`8786940e8b`, #20597)

## Usage

```bash
python experiments/sglang_fa_dispatch_dsv41_vllm_pcp_mla_dcp_kvcr_trtllm_helix_a2a_phase118/run.py --write
python experiments/sglang_fa_dispatch_dsv41_vllm_pcp_mla_dcp_kvcr_trtllm_helix_a2a_phase118/run.py --check
```
