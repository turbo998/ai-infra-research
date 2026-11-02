# vLLM Rust gRPC Readiness & Plain Embed Fast-Path, SGLang DeepSeek-V4.1 Indexer TP Shard & Large-Batch MHA KV Splits, TRT-LLM DEP4 MegaMoE & 64-Bit KDA Stride, and Dynamo Sticky Min-Load Routing (Phase 180)

Deterministic offline verification suite for Phase 180 (`priority: 189`, Track 2: `inference`).

## Audited Upstream Commits
- `vllm-project/vllm` @ `e8570ef757b6` (#65166), `f451c6254fe5` (#63870), `187f1a27dc7e` (#65059): Plain embedding input `WrapValidator` fast-path (`3.20x` speedup), Rust frontend gRPC `liveness` vs `readiness` separation (`0` false K8s restarts), and SM90+ guard on MoE top-k PDL launch attributes
- `sgl-project/sglang` @ `e33cd0c3a104` (#42968), `ad6ca42dea36` (#46082), `c35f89fc9a26` (#38719): DeepSeek-V4.1 attention-TP sharded dense prefill indexer (`75.0%` per-rank work reduction at TP4, `1.28x` speedup), `trtllm_mha` large-batch full-attention decode KV CTA splitting (`1.24x` speedup), and FP32 MoE router gate precision preservation (`0.0 ULP` truncation error)
- `NVIDIA/TensorRT-LLM` @ `4c2238ee524c` (#21225) & `bdcb577a93ea` (#21342): DeepSeek-V4-Pro DEP4 MegaMoE CuTe DSL bootstrap hang fix + oversized checkpoint RAM cache guard (`_can_cache_weights`), and Kimi-K3 KDA Small-BH prefill 64-bit state-pool row stride (`0` 32-bit overflows)
- `ai-dynamo/dynamo` @ `c260334d7048` (#15600) & `a908c1a14501` (#17700): Built-in `dynamo-sticky-min-load` session-sticky worker selection policy (`1.31x` session prefix locality) and offloading `SelectionLogFields` formatting from the scheduler actor to the routing task (`-35.0%` critical-path logging overhead)
