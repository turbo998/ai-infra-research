# Phase 121 Track B Experiment: SGLang Triton 3.9 Block Pointers, Ascend Ops MoE, Rust Sidecar/Abort, Ming RMSNorm, vLLM Zipf Routing & TRT-LLM Inductor/MiniMax DiT TP

Reproducible Python standard-library model (`run.py`) and sealed output (`summary.json`) covering 10 upstream commits from `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:

- `sgl-project/sglang@0b158f7` & `bacf65f`: `triton_block_ptr_compat.py` restoring `tl.make_block_ptr` / `tl.advance` on Triton 3.9 for FLA kernels and `ascend_ops` hybrid MoE A2A (`{"prefill":"alltoallv","decode":"low_latency"}`).
- `sgl-project/sglang@b7209cf`, `6a473b2`, & `4b43572`: Per-node `--sidecar` (`8 -> 1` sidecar per 8-GPU node), Rust `/generate` stream completion on scheduler abort frame (`CoreEvent::Finished`), and Ming-Omni DiT ordered RMSNorm at hidden width `3840` (`WARP_REDUCE_WIDTH = 4`).
- `vllm-project/vllm@b766f15`, `68eca9a`, & `6282c4c`: Anthropic inline-system probe `Exception` fallback, Rust frontend `vllm:num_requests_kv_fetch_by_stage` 3-stage gauges, and Zipf (`alpha=1.25`) MoE routing default (`20.1027x` hot-expert skew, `70.76%` top-8 mass share across 64 experts).
- `NVIDIA/TensorRT-LLM@c9d2dcc` & `a8ce107`: `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0` collective hang prevention and MiniMax-H3 DiT `ulysses_size * tp_size == world_size` 2D parallelism.

## Usage

```bash
python experiments/sglang_triton39_ascend_moe_rust_sidecar_ming_vllm_zipf_trtllm_phase121/run.py --check
```
