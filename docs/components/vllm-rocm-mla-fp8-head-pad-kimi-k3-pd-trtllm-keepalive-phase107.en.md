# vLLM ROCm AITER MLA FP8 Prefill Head-Padding Gate, Kimi-K3 1P1D Host DRAM Budget & TensorRT-LLM Silent-Stream Keepalive and Lightweight MPI Stop Analysis (Phase 107 · Priority 116)

> **中文版**: [vllm-rocm-mla-fp8-head-pad-kimi-k3-pd-trtllm-keepalive-phase107.md](vllm-rocm-mla-fp8-head-pad-kimi-k3-pd-trtllm-keepalive-phase107.md)  
> **Evidence Boundary**: `implementation` (first-hand upstream commits) + `own-offline` (deterministic stdlib modeling: `experiments/vllm_rocm_mla_fp8_head_pad_kimi_k3_pd_trtllm_keepalive_phase107/run.py --check`)

## 1. Primary Source Evidence & Implementation Paths

1. **`vllm-project/vllm@d0c20bb` (`#61863`: `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`)**:
   - **ROCm AITER MLA FP8 Prefill Head-Doubling Gate (`is_valid_fp8_prefill_num_heads`)**: The ROCm assembly FP8 MLA prefill kernel requires 16-aligned head counts per rank and replicate-pads smaller counts up to 16 before slicing the output back. For `1..8` heads/rank (e.g., Kimi-K2.x at TP8 with `8` heads/rank), padding to 16 at least doubles kernel work (`pad_multiplier = 2.0x`) plus pad/slice traffic (relative cost `11.2` vs unpadded BF16 cost `8.0`). `is_valid_fp8_prefill_num_heads` rejects `1..8` heads from FP8 prefill so they stay on unpadded BF16 prefill (**`1.40x` speedup** at 8 heads/rank), while keeping FP8 prefill for Kimi-K3 at TP8 (`12` heads/rank -> `1.3333x` pad ratio) and 16-aligned counts.
2. **`vllm-project/vllm@5f1a564` (`#63726`: `recipes/models/moonshotai/Kimi-K3.yaml`) & `7fe793a` (`#63769`), `984a418` (`#63763`)**:
   - **AMD MI355X Kimi-K3 1P1D Host DRAM Offload Budget Refresh**: Updates the 1P1D TP8/DCP8 MI355X disaggregated serving recipe (MoRIIO READ mode + `SimpleCPUOffloadConnector` on prefill) from `1,799 GB` to **`2,399 GB` (`299.875 GB/rank`)** host DRAM KV offload budget, increasing 64K-context cached sequence capacity by **`1.3342x` (`393 -> 524` sequences)**; also adds `--enable-log-outputs`/`--enable-log-deltas` and preserves engine logprob ordering in the Rust frontend.
3. **`NVIDIA/TensorRT-LLM@90e573e` (`#21061`: `tensorrt_llm/serve/responses_utils.py`) & `df95eda` (`#20434`: `tensorrt_llm/llmapi/_llmapi_mpi_stop.py`) & `b6751ca` (`#21097`)**:
   - **Responses API `15.0s` Silent-Stream Keepalive & Standalone ZMQ MPI Stop Helper**: `keep_responses_stream_alive` re-emits `event: response.in_progress` every **`15.0s`** (`RESPONSES_STREAM_KEEPALIVE_INTERVAL_S = 15.0`) while a stream is silent during queueing, prefill, or multi-step tool-call generation (emitting `6` keepalives across a `95s` silent window and preventing proxy idle disconnects); meanwhile `_llmapi_mpi_stop.py` sends authenticated ZMQ shutdown frames using only `pyzmq` without importing `tensorrt_llm` or initializing MPI (cutting helper cold startup from `14.5s` to **`0.18s`, `80.56x` faster**).
4. **`sgl-project/sglang@4549937` (`#44487`), `d3a157a` (`#46620`), `f9d132b` (`#31286`), `e4d4a9a` (`#46536`)**:
   - Preserves nested Puzzle configurations in Nano-Nemotron-VL, removes unused `CacheInitParams::cache_ttl_second` in HiCache, eliminates startup TorchScript deprecations, and guards CUDA-only RoPE paths in diffusion models.

## 2. Architect Design Exercises & Falsifiable Questions

- **Calculation Exercise**: With unpadded BF16 prefill cost `1.0/head`, 16-aligned FP8 prefill kernel cost `0.62/head`, and pad+slice overhead `0.08 * padded_heads`, verify why `num_heads = 8` padded to 16 costs `11.2 > 8.0` (favoring BF16 by `1.40x`) whereas `num_heads = 12` padded to 16 costs `11.2 < 12.0` (favoring FP8).
- **Falsifiable Hypothesis**: Without SSE keepalives, any tool-call generation or long-context prefill silent for `> 60s` behind a standard reverse proxy suffers deterministic stream termination before the first output frame arrives (prevented by TensorRT-LLM's `15.0s` `response.in_progress` keepalive interval).
