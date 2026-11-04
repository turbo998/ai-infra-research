# Distributed Training, Collectives & KV Fabrics: Megatron-LM FlashInfer MXFP8/NVFP4 RL Train-Gen Bitwise Parity, TE Compiled MoE Router & Zero-Copy Packed-QKV Backward, PyTorch Inductor Wide-Broadcast Realization, RCCL `gfx1250` Batched-Poll DDA LL & Mooncake Multi-Protocol Memory Ownership (Phase 182)

> 🌐 **Language / 语言切换**: **[中文](distributed-training-megatron-moe-rl-parity-te-compiled-router-inductor-broadcast-rccl-gfx1250-dda-mooncake-hip-ipc-phase182.md)** | **[English](distributed-training-megatron-moe-rl-parity-te-compiled-router-inductor-broadcast-rccl-gfx1250-dda-mooncake-hip-ipc-phase182.en.md)**

---

## 1. Core Architectural Problems & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 182 (`priority: 191`) audits **14 upstream mainline commits** across `NVIDIA/Megatron-LM`, `NVIDIA/TransformerEngine`, `pytorch/pytorch`, `ROCm/rocm-systems` (RCCL), `kvcache-ai/Mooncake`, `LMCache/LMCache`, and `ai-dynamo/nixl`:

1. **Megatron-LM FlashInfer MXFP8 / Per-Token NVFP4 RL Train-Gen Bitwise Parity, Fused Batch-Invariant Squared-ReLU + MXFP8 Quantization & GRPO Observer Preservation**:
   - `ebcf9d67221d` (#7798) introduces `moe_training_matches_inference: true` for RL post-training: under `inference_optimized`, `batch_invariant_mode`, `moe_token_dispatcher_type='alltoall'`, and selective `moe` recompute, the training value pass and `eval()` log-prob forward execute the exact same FlashInfer routed MXFP8 or per-token NVFP4 kernel as generation (with `unpermute_with_bf16_rank_partials` summing routes in generation's top-k order via FP32 FMAs before BF16 quantization), while backward uses the unchanged Transformer Engine (TE) recompute pass ($d_{\mathrm{FI}} \le 1.004 \cdot d_{\mathrm{TE}}$ relative distance to BF16). On 4x GB300 (`EP=4`, Nano-v3 128-expert top-6):
     - **Per-token NVFP4**: achieves **bitwise train–gen log-prob parity (`0` ULPs mismatch)**, accelerates log-prob forward by **`3.37x–4.357x`** (`23.53 -> 5.40 ms` at `4,096` tok/rank by replacing TE's ~700 per-expert Python kernel launches), accelerates the full train step by **`2.28x–2.897x`** (`59.47 -> 20.53 ms` at `1,024` tok/rank, and `1.05x–1.21x` faster than BF16 TE + recompute), and saves **`178 MiB`** of static memory per layer per rank (`1,543 vs 1,721 MiB`).
     - **MXFP8**: achieves bitwise log-prob parity, **`1.40x–1.422x`** faster log-prob forward (`9.27 -> 6.52 ms` at `4,096` tok/rank), and **`1.13x–1.151x`** faster train step (`31.43 -> 27.30 ms` at `1,024` tok/rank).
   - `4a4064f0631c` (#8645) fuses `squared_relu_with_probs` and `mxfp8_quantize_rows` into `squared_relu_with_probs_quantize_mxfp8` under batch-invariant mode, saving `0.07–0.13 ms/step` on GB200 (`Nemotron-3-Nano`); `993bf83ec875` (#8850) keeps the saved-tensor observer enabled when `create_cudagraphs()` is called before training graphs are recorded in GRPO reference/old log-prob passes.
2. **TransformerEngine Compiled Fused MoE Routing & Zero-Copy Packed-QKV Backward, PyTorch Inductor Wide-Broadcast Realization & AOTI C-Shim CUDA Graph Capture**:
   - `720b25258be3` (#3789) replaces `Tensor.select` with `SplitAlongDim` for packed `qkv_layer`/`kv_layer` eager inputs in TE MultiheadAttention, eliminating full-shape zero-filled per-view gradient tensor materializations in backward (**`0` zero-filled view-gradient allocations**); `2d65f35b662c` (#3691) adds `torch.compile(fullgraph=True)` support for fused MoE routing and quantile-balancing histograms via `setup_context` (`3,391` router tests pass).
   - `1933a0a9ee51` (#202772) applies `realize_reads_threshold` in PyTorch Inductor once a broadcast is at least `realize_broadcast_reuse_threshold = 64` wide, preventing per-element inlining of row-wise router weights and cutting the `gpt-oss` MoE down-projection epilogue on B200 from `36.4 us` to **`16.8 us` (`2.167x` speedup, `-53.85%`)**.
   - `b31d68db9561` (#202227) exposes `at::cuda::MemPool`, `at::cuda::CUDAGraph`, and `aoti_torch__foreach_copy_` in the PyTorch AOTI C shim with thread-local capture and `cudaMallocAsync` rejection.
3. **ROCm RCCL `gfx1250` Batched-Poll Prefetch DDA LL AllReduce & ReduceScatter Tier Caps**:
   - `f59fd15cb79a` (#14880) prefetches initial send packets into L2 before the epoch barrier (`ddaLLPrefetchSend`) and batches peer-line polling up to **`8` lines** under a single load wait (`ddaLLWaitLineFlagsBatched`) in the `gfx1250` (`MI455X`) DDA LL one-shot AllReduce.
   - `3b0c0acddf20` (#14759) raises `gfx1250` SPX ReduceScatter DDA LL128 caps to **`224 MiB`** total below 8 ranks and **`7.5 MiB/rank`** from 8 ranks (clipped at `256 MiB`), and raises DDA LL caps to `4 MiB` total (`< 8` ranks) and **`1.0 MiB/rank`** (`>= 8` ranks), avoiding premature fallback to slower Ring kernels across `32M–256M` sizes on 4–64 GPUs.
4. **Mooncake Multi-Protocol Memory Ownership & Redis Auto-Reconnect, LMCache `RawBlock` Queue Bounding & NIXL Zero-Alloc NVTX / EFA Gating**:
   - `748af1a53dba` (#3548) registers HIP IPC memory at `hipMemGetAddressRange` allocation-base granularity in Mooncake TransferEngine with address-keyed reference counting and protocol-scoped descriptor removal across **`3` transport protocols** (`"hip"`, `"rdma"`, `"tcp"`), fixing cross-process sub-allocation relocation and `rdma+hip` base-address collisions on 8x MI308X / MI355X; `4704c67f13ff` (#4042) adds **`3 s`** connect/command timeouts, TCP keepalive, and automatic reconnect with `AUTH`/`SELECT` replay to `RedisStoragePlugin` after idle load-balancer disconnects (`~4000s`).
   - `338d5fba394e` (#4697) reserves completion-queue (CQ) capacity for every request in LMCache `RawBlock`'s in-flight map until reaped (`0` CQ overflow panics); `e2cd0283d5df` (#2561) loads the NIXL UCX EFA plugin only when EFA devices are detected (fixing exit code `139 SIGSEGV -> 0`); `13c52d21bdc6` (#2571) resolves NVTX record names via a **64**-slot compile-time FNV-1a table and 64B stack buffer, saving **`6` heap allocations/iteration** in `nixlbench` (`92 -> 86` sampled, `42 -> 38` unsampled).

---

## 2. End-to-End Dataflow & Architecture Topology (Pillar 1)

```
+--------------------------------------------------------------------------------------------------------+
| Phase 182 Distributed Training, Collectives & KV Fabric: Megatron-LM, TE, PyTorch, RCCL & Mooncake/NIXL|
+--------------------------------------------------------------------------------------------------------+
  [Megatron-LM RL Train-Gen Bitwise Parity & Fused Batch-Invariant Quant (#7798, #8645, #8850)]
    |-- FlashInfer RL Value/LogProb (#7798)--> Bitwise Parity (0 ULPs): NVFP4 LogProb 4.357x, Train 2.897x (-178MiB)
    |-- Fused SqReLU+MXFP8 Quant (#8645)   --> squared_relu_with_probs_quantize_mxfp8 (-0.07..0.13 ms/step GB200)
    +-- GRPO Saved-Tensor Observer (#8850) --> Preserve Observer Across Empty create_cudagraphs() in Old LogProb
                                           |
                                           v
  [TransformerEngine, PyTorch Inductor/AOTI & ROCm RCCL gfx1250 (#3789, #3691, #202772, #202227, #14880, #14759)]
    |-- TE SplitAlongDim & Router (#3789)  --> 0 Zero-Filled Packed-QKV Grad Views + Fullgraph Compiled MoE Router
    |-- PyTorch Inductor & AOTI (#202772)  --> Wide-Broadcast (>=64) Producer Realize (2.167x B200) + C-Shim Graph
    +-- RCCL gfx1250 DDA LL/LL128 (#14880) --> L2 Send Prefetch + 8-Line Batched Poll + RS 224MiB/7.5MiB Tier Caps
                                           |
                                           v
  [Mooncake HIP IPC / Redis Recovery, LMCache RawBlock & NIXL UCX/NVTX (#3548, #4042, #4697, #2561, #2571)]
    |-- Mooncake Multi-Transport (#3548)   --> Base-Granular HIP IPC + 3-Protocol Ownership + Redis 3s Reconnect
    +-- LMCache & NIXL Hardening (#4697)   --> In-Flight CQ Reservation + EFA Device Gate + 64-Slot FNV-1a NVTX
```

---

## 3. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/megatron_moe_rl_parity_rccl_gfx1250_dda_mooncake_hip_ipc_phase182/run.py`](../../experiments/megatron_moe_rl_parity_rccl_gfx1250_dda_mooncake_hip_ipc_phase182/run.py) `--check`:
- `megatron_nvfp4_logprob_max_speedup = 4.357`, `megatron_nvfp4_train_max_speedup = 2.897`, `megatron_nvfp4_static_mem_saved_mib = 178`
- `megatron_mxfp8_logprob_max_speedup = 1.422`, `megatron_train_gen_bitwise_logprob_mismatch_ulps = 0`
- `pytorch_inductor_wide_broadcast_speedup = 2.167` (`36.4 -> 16.8 us`), `te_packed_qkv_zero_fill_grad_allocations = 0`
- `rccl_gfx1250_dda_ll_batched_poll_lines = 8`, `rccl_gfx1250_rs_ll128_sub8rank_cap_mib = 224.0`, `rccl_gfx1250_rs_ll128_per_rank_cap_mib = 7.5`
- `mooncake_transport_ownership_protocols = 3`, `mooncake_redis_reconnect_timeout_s = 3`, `lmcache_rawblock_cq_overflow_panics = 0`, `nixl_nvtx_allocs_saved_per_iter = 6`

---

## 4. Enterprise Business Scenario Playbook (Pillar 4)

| Scenario Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **Blackwell Large-Scale MoE Reinforcement Learning (RL / GRPO)** | Enable `moe_training_matches_inference: true` on GB200/GB300 RL runs using FlashInfer routed NVFP4 or MXFP8 rollouts (`#7798`), eliminating importance-sampling ratio bias while achieving `4.357x` NVFP4 log-prob and `2.897x` train-step speedups. | Enabling `moe_training_matches_inference` when training and generation use different EP sizes, `ETP > 1`, or flex (`DeepEP`/`HybridEP`) dispatchers (explicitly rejected by config validation). |
| **MI455X (`gfx1250`) Collectives & PyTorch/TE Compilation** | Use tuned RCCL ReduceScatter DDA LL (`1 MiB/rank`) and LL128 (`7.5 MiB/rank`) caps plus 8-line batched-poll AllReduce on 4–64 MI455X GPUs (`#14759/#14880`), and Inductor wide-broadcast realization on B200 (`#202772`, `2.167x`). | Creating AOTI C-shim `MemPool` instances under `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` (`#202227` explicitly rejects this backend). |
| **MI308X / MI355X Cross-Process KV Transfer & Long-Lived Redis Metadata** | Deploy Mooncake base-granular HIP IPC registration with protocol-scoped (`hip`/`rdma`/`tcp`) descriptor removal (`#3548`) and 3s timeout + auto-reconnect for Redis metadata behind TCP load balancers (`#4042`). | Admitting LMCache `RawBlock` batches based solely on submission-queue length without reserving completion-queue capacity across in-flight requests (`#4697`). |
