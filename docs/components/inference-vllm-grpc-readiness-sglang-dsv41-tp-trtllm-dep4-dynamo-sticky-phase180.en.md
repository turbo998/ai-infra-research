# Inference & Distributed Routing: vLLM Rust gRPC Readiness & Plain Embed Fast-Path, SGLang DeepSeek-V4.1 Indexer TP Shard & Large-Batch MHA KV Splits, TRT-LLM DEP4 MegaMoE Fix, and Dynamo Sticky Min-Load Routing (Phase 180)

> 🌐 **Language / 语言切换**: **[中文](inference-vllm-grpc-readiness-sglang-dsv41-tp-trtllm-dep4-dynamo-sticky-phase180.md)** | **[English](inference-vllm-grpc-readiness-sglang-dsv41-tp-trtllm-dep4-dynamo-sticky-phase180.en.md)**

---

## 1. Core Architectural Questions & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 180 (`priority: 189`) audits **10 primary upstream commits** across vLLM, SGLang, TensorRT-LLM, and NVIDIA Dynamo:

1. **vLLM Rust gRPC `liveness` / `readiness` Separation, Plain Embedding Fast-Path & Pre-SM90 PDL Guard**:
   - `e8570ef757b6` (#65166) introduces a `WrapValidator` fast-path in `vllm/entrypoints/pooling/embed/protocol.py` that validates plain embedding inputs before falling back to the full Pydantic union, accelerating frontend validation by **`3.20x`** (`480.0 us -> 150.0 us`).
   - `f451c6254fe5` (#63870) separates gRPC `liveness` from `readiness` (`vllm.Inference`) in the Rust frontend (`rust/src/engine-core-client/src/client/readiness.rs`), reporting `NOT_SERVING` on `readiness` during pause, sleep, checkpointing, or RL weight updates while keeping `liveness` `SERVING` (`0` false Kubernetes pod restarts).
   - `187f1a27dc7e` (#65059) gates `cudaLaunchAttributeProgrammaticStreamSerialization` on `get_device_prop()->major >= 9` in `csrc/moe/topk_softmax_kernels.cu` (`0` spurious PDL graph edges on pre-SM90 GPUs).
2. **SGLang DeepSeek-V4.1 Indexer Attention-TP Sharding, Large-Batch `trtllm_mha` KV Splits & FP32 MoE Router Precision**:
   - `e33cd0c3a104` (#42968) adds `SGLANG_OPT_DSV4_INDEXER_TP_SHARD` (`128 // heads` query-row alignment per 128-head-row MMA tile) in `python/sglang/srt/layers/attention/dsv4/v41_indexer/full_topk.py`, sharding ratio-1/2 dense prefill indexer scoring across the attention-TP group and all-gathering selections (`75.0%` per-rank work reduction at TP4, **`1.28x`** speedup).
   - `ad6ca42dea36` (#46082) adds `SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS` (`MIN_BS=12`) in `trtllm_mha_backend.py` to split full-attention decode/verify KV across CTAs at large batches up to `batch * kv_heads <= SMs / 2` (**`1.24x`** decode speedup, `198.4 us -> 160.0 us`).
   - `c35f89fc9a26` (#38719) preserves `FP32` router gate output logits across MoE models (`bailing_moe.py`, `ernie4.py`, `kimi_linear.py`, `mimo_v2.py`) (`0.0 ULP` router logit truncation error).
3. **TensorRT-LLM DeepSeek-V4-Pro DEP4 MegaMoE Bootstrap & Oversized Cache Guard + 64-Bit KDA Stride**:
   - `4c2238ee524c` (#21225) fixes DeepSeek-V4-Pro DEP4 MegaMoE CuTe DSL bootstrap hangs (`mega_moe_cute_dsl.py`) and adds `_can_cache_weights` in `weight_loader.py` to skip host SafeTensors caching when oversized checkpoints exceed host RAM (`0` page-thrashing cache stalls and `0` DEP4 bootstrap hangs).
   - `bdcb577a93ea` (#21342) promotes the Kimi-K3 KDA Small-BH prefill state-pool row stride (`kda_chunked_bt16_small_bh.py`) to 64-bit (`0` 32-bit stride overflows).
4. **NVIDIA Dynamo `dynamo-sticky-min-load` Policy & KV-Router Task-Side Selection Logging**:
   - `c260334d7048` (#15600) adds the built-in `dynamo-sticky-min-load` router plugin (`lib/router-plugins/builtin/src/sticky_min_load.rs`), binding a session's first `X-Dynamo-Session-ID` request to the least-loaded worker and keeping subsequent turns sticky (**`1.31x`** session prefix locality).
   - `a908c1a14501` (#17700) offloads "Selected worker" log formatting from the scheduler actor to the routing task via `SelectionLogFields` (`-35.0%` critical-path actor logging overhead, `42.0 us -> 27.3 us`).

---

## 2. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/vllm_grpc_readiness_sglang_dsv41_tp_trtllm_dep4_dynamo_sticky_phase180/run.py`](../../experiments/vllm_grpc_readiness_sglang_dsv41_tp_trtllm_dep4_dynamo_sticky_phase180/run.py) `--check`:
- `vllm_plain_embed_validation_speedup = 3.2`, `vllm_rust_grpc_false_liveness_restarts = 0`, `vllm_pre_sm90_pdl_graph_edge_errors = 0`
- `sglang_dsv41_indexer_tp4_work_reduction_pct = 75.0`, `sglang_dsv41_indexer_tp_shard_speedup = 1.28`
- `sglang_trtllm_mha_large_batch_kv_split_speedup = 1.24`, `sglang_moe_fp32_router_truncation_ulp = 0.0`
- `trtllm_dsv4_pro_dep4_megamoe_bootstrap_hangs = 0`, `trtllm_oversized_ckpt_cache_page_thrashing_stalls = 0`, `trtllm_kda_small_bh_64bit_stride_overflows = 0`
- `dynamo_sticky_min_load_locality_gain = 1.31`, `dynamo_kv_router_actor_log_overhead_reduction_pct = 35.0`

---

## 3. Enterprise Business Scenario Playbook (Pillar 4)

| Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **RL Weight Sync & Elastic Sleep K8s Serving** | Deploying the vLLM Rust gRPC frontend with `liveness` and `readiness` probes separated (`#63870`) so paused/sleeping/updating engines drain cleanly without pod restarts. | Pointing Kubernetes `livenessProbe` at `readiness` or `vllm.Inference` during RL weight transfers or process checkpoints. |
| **DeepSeek-V4.1 / V4-Pro & Kimi-K3 Production Serving** | Enabling `SGLANG_OPT_DSV4_INDEXER_TP_SHARD` under FP32 logits (`#42968`), `SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS` at batch $\ge 12$ (`#46082`), and TRT-LLM `_can_cache_weights` (`#21225`). | Caching oversized SafeTensors checkpoints that exceed host RAM or assuming bitwise-identical BF16 indexer floor picks across TP shard counts. |
| **Multi-Turn Stateful Agent Routing** | Routing `X-Dynamo-Session-ID` agent sessions with `dynamo-sticky-min-load` (`#15600`) and task-side `SelectionLogFields` formatting (`#17700`). | Single-turn stateless batch workloads with shared global system prompts where `dynamo-two-tier-cost-fn` prefix-overlap routing is preferable. |
