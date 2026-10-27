# SGLang DSpark Simulated Acceptance & HiCache PD Early-Send Guard, DeepEP v2 Masked GEMM Buffer Shrinking, vLLM ROCm Sparse MLA Gate, TensorRT-LLM MPI Pigeonhole Barrier & AI Infra Innovators Comparison (Phase 103)

- **Question ID**: `sglang-dspark-hicache-deepep-vllm-trtllm-innovators-phase103` (`priority: 112`, component: `inference`)
- **Evidence Status**: `own-offline` (First-hand upstream commit inspection + deterministic Python 3.12 stdlib model `experiments/sglang_dspark_hicache_vllm_trtllm_phase103/run.py`)
- **Chinese Companion**: [`sglang-dspark-hicache-deepep-vllm-trtllm-innovators-phase103.md`](sglang-dspark-hicache-deepep-vllm-trtllm-innovators-phase103.md)

## 1. Primary Upstream Sources (Zero-Overlap)

1. **SGLang `3927a73a34` (#44778)**: `[Fix] kda: Use raw beta in prefill CUDA graphs`
2. **SGLang `24d55621a3` (#46404)**: `[Fix] dspark: Draw the simulated-acceptance bonus at the simulated row`
3. **SGLang `2a7987f900` (#44081)**: `[Fix] PD: Skip the cached-prefix early send in batches that start a HiCache load-back`
4. **SGLang `a09c1a096c` (#46309)**: `[Perf] deepep_v2: Reduce masked GEMM buffer memory during CUDA Graph capture`
5. **vLLM `939ff73503` (#63785)**: `[Perf] Drop BasevLLMParameter's Python __torch_function__`
6. **vLLM `8d88bc5d81` (#60154)**: `[ROCm][DSv4/4.1] Fuse inverse RoPE and MXFP8 quant into the aiter sparse MLA and sparse MLA gating changes`
7. **vLLM `f8f7634cb4` (#63342)**: `[ROCm][Perf][Kimi-K3] Use block_m=128 for the fused KDA prefill conv1d`
8. **TensorRT-LLM `8bd66f6a71` (#20904)**: `[None][fix] Explain K/V pool coalescing failures; test Qwen3.8-Flash-Next QSA layouts and FP8/NVFP4 KV`
9. **TensorRT-LLM `0ba9f945f0` (#20996)**: `[TRTLLM-15765][refactor] Remove the ModelEngine ABC and settle the executor interface`
10. **TensorRT-LLM `105401ea8d` (#20467)**: `[https://nvbugs/6932260][fix] Ensure one-task-per-rank for MpiSession.submit`

---

## 2. Core Implementation Mechanics & Quantitative Analysis

### 2.1 SGLang DeepEP v2 32.0x Masked GEMM Slab Reduction During CUDA Graph Capture (`a09c1a096c`)

During CUDA Graph capture without attention DP/CP (`attn_dp_size == 1, attn_cp_size == 1, tp_size == ep_group_size`), TP ranks share a batch (or `tensor_split` shards differing by at most 1 row), and each token selects a given expert at most once. `_masked_token_capacity` bounds `masked_max_m` to:
$$\text{masked\_max\_m} = \min\!\left(\text{max\_dispatch} \times \text{ep\_size},\; \left\lceil \frac{\max(1, \text{local\_tokens} \times \text{ep\_size} + \text{ep\_size} - 1)}{128} \right\rceil \times 128\right)$$
For `max_dispatch=512, ep_size=8` (32 local experts, `H=7168` BF16), `masked_max_m` at `local_tokens=1..15` drops from `4,096` rows (**1,792.0 MiB**) to `128` rows (**56.0 MiB**), a **32.0x (96.88%)** reduction saving **1,736.0 MiB** while keeping `ElasticBuffer` and collective handles unchanged.

### 2.2 SGLang PD HiCache Load-Back Early-Send Gate (`2a7987f900`) & DSpark Simulated Bonus (`24d55621a3`)

- **PD HiCache Early-Send Hazard Gate (`2a7987f900`)**: Sets `load_pending = batch.hicache_consumer_index >= 0` and passes `allow_send=not load_pending` to `maybe_send_cached_prefix_chunk`, preventing disaggregated prefill from RDMA-sending device prefix pages that an asynchronous HiCache load-back in the same batch is still writing (`2 → 0` corrupted early-sends).
- **DSpark Simulated-Acceptance Bonus (`24d55621a3`)**: Loads `correct = tl.load(sim_ptr)` under `HAS_SIM` in `_static_accept_tail_kernel` so the bonus token is drawn at the simulated acceptance row (`row = 4`) rather than the greedy match row (`row = 1`) with the request's sampling parameters.

### 2.3 vLLM Parameter Dispatch & ROCm Sparse MLA (`939ff73503`, `8d88bc5d81`, `f8f7634cb4`) & TensorRT-LLM (`8bd66f6a71`, `105401ea8d`)

- **vLLM (`939ff73503`, `8d88bc5d81`, `f8f7634cb4`)**: Removes `BasevLLMParameter.__torch_function__` (`1 → 0` Python interpreter entries per op); gates ROCm `aiter` sparse MLA prefill at `_AITER_SPARSE_MLA_PREFILL_MAX_HEADS = 64` (using gather + OPUS at `128` heads/rank while keeping fused inverse RoPE + MXFP8 decode at all head counts); and sets Kimi-K3 fused KDA prefill conv1d `PREFILL_CONV_BLOCK_M = 128`.
- **TensorRT-LLM (`8bd66f6a71`, `105401ea8d`)**: Validates Qwen3.8-Flash-Next on GB300 (`95.49%` GSM8K / `87.33%` MMLU with FP8 KV; `95.26%` GSM8K / `87.57%` MMLU with NVFP4 KV + MTP) and enforces an entry MPI barrier in `MpiSession.submit` so each worker takes exactly 1 task (`[1, 1, 1, 1]`, pigeonhole principle) without a trailing barrier.

### 2.4 Quantitative Comparison of AI Infra Innovators (Fireworks AI, Together AI, Cerebras WSE-3, Groq LPU vs Blackwell GB300)

On a 70B-active-equivalent low-latency decode benchmark (`35 GB` NVFP4 / `70 GB` FP8 active weights):
- **Together AI Turbo Speculative (`3,969.1 tok/s`)** & **NVIDIA GB300 NVL72 (`3,790.4 tok/s`)** & **Fireworks AI FireAttention PD (`3,452.0 tok/s`)** amortize `64 TB/s` TP8 HBM3e weight reads across `3.2–3.6` accepted tokens per verification step while preserving TB-scale KV cache capacity and HiCache prefix reuse.
- **Cerebras WSE-3 (`44 GB` on-wafer SRAM, `~21 PB/s`, `2,369.7 tok/s` at `1.0` tok/step)** and **Groq LPU (`230 MB/chip` SRAM, `80 TB/s/chip`, `1,169.6 tok/s`)** eliminate the single-token HBM bandwidth wall via on-chip SRAM, making WSE-3 an exceptional ultra-fast speculative draft generator for GPU verifiers, while facing multi-chip/wafer capacity boundaries for long-context KV caches.
