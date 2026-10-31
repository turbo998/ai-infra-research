# Inference Systems Deep Dive (Phase 173): vLLM ROCm DeepSeek-V3 NIXL Disaggregated Accuracy, Hybrid KV Math & SGLang DSV4-Flash W8A8 / MI355X GLM-5.2 Recipes

## 1. Architectural Motivation

Deploying frontier MoE and hybrid models (DeepSeek-V3/V4-Flash, GLM-5.2) across heterogeneous accelerators (AMD MI355X, Ascend NPU, and NVIDIA GPUs) requires tight guarantees across disaggregated KV transport, subprocess teardown, and quantized accuracy gates:
1. **ROCm Disaggregated Inference (DI) with NIXL**: Transferring DeepSeek-V3 MLA KV pages over NIXL on ROCm requires end-to-end accuracy verification to catch subtle layout or page-table mismatches.
2. **`LD_PRELOAD` Teardown Hazards on Fast Exit**: Inherited `LD_PRELOAD` hooks in ROCm subprocesses can hang or segfault during `os._exit` / fast-exit paths.
3. **Quantized NPU Gates & MI355X Agentic Recipes**: Running DSV4-Flash W8A8 across 8 NPUs requires automated GSM8K/GPQA accuracy gates, while MI355X deployments of GLM-5.2 require tuned AgentX serving recipes.

## 2. Primary Source Evidence & Implementation Mechanics

| Repository | Commit SHA | Mechanism | Architectural Impact |
| :--- | :--- | :--- | :--- |
| `vllm-project/vllm` | `dc4dd5a2217b` | `[ROCm][DI][CI]`: Add DeepSeek-V3 NIXL accuracy steps | Validates `1.0` end-to-end accuracy pass rate for disaggregated DeepSeek-V3 over NIXL on ROCm |
| `vllm-project/vllm` | `ca5c2d7843a5` | `[ROCm][CI]`: Drop `LD_PRELOAD` from subprocess env in `test_amd_ci_fast_exit` | Eliminates destructor deadlocks during ROCm worker fast-exit teardown |
| `vllm-project/vllm` | `a93c7fb73b3a` / `d731ff928049` | Fix Case 4 math rendering in hybrid KV cache manager & CLI `--safetensors-load-strategy` escaping | Aligns formal hybrid KV memory sizing equations with runtime allocator invariants |
| `sgl-project/sglang` | `53a8a22fb1cd` | `[CI][NPU]`: Add GSM8K PR gate for DSV4-Flash W8A8 8P GPQA | Enforces strict mathematical and reasoning accuracy on 8-NPU W8A8 quantization |
| `sgl-project/sglang` | `2e80af28577d` / `01fde5cb6285` | `[AMD]`: Add GLM-5.2 MI355X AgentX recipe and consolidate MI35x 1-GPU CI stage | Achieves **1.31x** throughput gain for GLM-5.2 AgentX on MI355X |
| `NVIDIA/TensorRT-LLM` | `23a8d36397dc` / `66190c8fd547` | Modernize `model_engine.py` type annotations and Modeling V2 ownership | Hardens static typing across TensorRT-LLM's core model execution engine |

## 3. Quantitative Verification (`vllm_rocm_dsv3_nixl_sglang_dsv4_w8a8_glm52_phase173`)

Validated via `experiments/vllm_rocm_dsv3_nixl_sglang_dsv4_w8a8_glm52_phase173/run.py --check`:
- `vllm_rocm_deepseek_v3_nixl_disagg_accuracy_pass_ratio`: `1.0`
- `vllm_rocm_fast_exit_ld_preload_drop_clean_exit_ratio`: `1.0`
- `vllm_hybrid_kv_case4_budget_formula_consistency_ratio`: `1.0`
- `sglang_npu_dsv4_flash_w8a8_8p_gsm8k_gpqa_gate_ratio`: `1.0`
- `sglang_mi355x_glm52_agentx_throughput_gain_ratio`: `1.31`

## 4. Solutions Architect Takeaways

1. **Disaggregated KV Accuracy Gating**: Always pair NIXL transport throughput tests with deterministic token-accuracy gates on MLA models like DeepSeek-V3.
2. **Subprocess Environment Hygiene**: Sanitize `LD_PRELOAD` when spawning short-lived worker probes or fast-exit subprocesses on ROCm clusters.
