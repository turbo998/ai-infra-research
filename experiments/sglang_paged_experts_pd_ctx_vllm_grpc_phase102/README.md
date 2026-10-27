# Phase 102 Track B: SGLang Paged-Experts Fused Slot Lookup & PD Context Isolation, vLLM Rust gRPC Detok Bypass & TRT-LLM OpenEngine Global Transfer IDs

Deterministic Python 3.12 standard-library model reproducing:
- **SGLang `paged_experts_decide` Fusion (`8063296528`, #46457)**: Fuses `run_ids` and `run_w` slot/weight writes directly into the single-warp `paged_experts_decide_kernel`, cutting per-layer helper launches by **66.67%** (`6 -> 2` launches/layer, saving `244` launches across 61 MoE layers).
- **SGLang PD Worker Device Binding (`7fe5b127e4`, #45678)**: `bind_thread_to_device(gpu_id)` in Mooncake and MoRI transfer workers eliminates **7** stray primary CUDA contexts on `GPU 0` (`2,940.0 MiB` saved on an 8-GPU node).
- **vLLM Rust Frontend (`2fb5111bba`, `803bc13cf9`, `4216a234cf`, `447d28eef2`, `40fe396b60`)**: Skips detokenization when `output_text=false` (`60.621 ms` CPU saved per 32K tokens) and selectively rewrites typed `--reasoning-effort-map`.
- **TensorRT-LLM OpenEngine Disaggregated Prefill (`2d3e60e27e`, #21043)**: Replaces colliding local `ctx_request_id` (`274 == 274`) with `MIN_GLOBAL_ID + secrets.randbelow((1 << 63) - MIN_GLOBAL_ID)`.

## Usage
```bash
python experiments/sglang_paged_experts_pd_ctx_vllm_grpc_phase102/run.py
python experiments/sglang_paged_experts_pd_ctx_vllm_grpc_phase102/run.py --check
```
