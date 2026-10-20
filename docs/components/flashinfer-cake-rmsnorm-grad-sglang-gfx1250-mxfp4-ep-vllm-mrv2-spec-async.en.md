# FlashInfer Cake RMSNorm Cooperative Backward Ring (`SM100`/`SM103`/`SM107`), SGLang `gfx1250` W4A8 MXFP4 MoE Sentinel EP Routing & vLLM `MRV2` Async Speculative Scheduling

> **中文版**: [flashinfer-cake-rmsnorm-grad-sglang-gfx1250-mxfp4-ep-vllm-mrv2-spec-async.md](flashinfer-cake-rmsnorm-grad-sglang-gfx1250-mxfp4-ep-vllm-mrv2-spec-async.md)  
> **Evidence Class**: `implementation` (verified upstream commits) + `own-offline` (`experiments/flashinfer_cake_rmsnorm_sglang_ep_vllm_mrv2/run.py --check`)

## 1. First-Hand Source Verification & Architectural Mechanics

1. **FlashInfer Cake RMSNorm `H1536`/`H4096` Cooperative Persistent Backward Ring (`flashinfer-ai/flashinfer` commit `0d062af33b6b42786bc5442acb802dd265242355`, `flashinfer/cake_rmsnorm_train_backend/cake_jit.py` & `build.json`)**:
   - Upgrades plain RMSNorm backward (`dx`, `dweight`) on `sm_100a` (B200), `sm_103a` (GB300), and `sm_107a` (R200):
     - Routes to cooperative persistent kernels (`cooperative=True`, `ctas_per_sm=1`, `grid_factor=2`) when `T > 1024` (`H=1536`) or `T > 256` (`H=4096`) on `sm_100a`/`sm_103a`, and when `T > 2048` (`H=1536`) or `T > 1024` (`H=4096`) on `sm_107a`;
   - Stages `dy` and `x` rows through a `cp.async.bulk` + `mbarrier` shared-memory ring while preserving deterministic fixed-order FP32 `dweight` reduction across `partial` and `counters`, achieving **`1.1245x`** backward speedup on B200 (`H=1536, T=65536`: `113.53 -> 100.96 us`) and **`1.2161x`** on R200 (`H=4096, T=65536`: `172.90 -> 142.18 us`, `author-benchmark`).
2. **SGLang AMD `gfx1250` GPT-OSS W4A8 MXFP4 MoE Expert Parallelism (`sgl-project/sglang` commit `89860fbd2e3cf460c0968435670b7fdef6ab5529`, `python/sglang/srt/layers/moe/fused_moe_triton/aiter_mxfp4_w4a8_moe.py`)**:
   - `build_ep_expert_map` constructs a `(num_experts + 1,)` int32 lookup table initialized to `-1` with `[start : start + num_local_experts] = 0..num_local_experts-1`;
   - `_ep_routing` computes `block_m` from the global expert count `n_expts_tot`, routes non-local tokens to a sentinel bin, clamps `gather_src.clamp_(0, hidden_states.shape[0] - 1)` before host gather to prevent out-of-bounds reads on unwritten sentinel rows, and passes `gate_valid` into GEMM2 `moe_gemm_a8w4` prior to cross-rank AllReduce.
3. **vLLM Model Runner V2 (`MRV2`) Speculative Decoding Async Scheduling Guard (`vllm-project/vllm` commit `f1dbfef79973d0f8d05be56e8614db0409e3d7d3`, `vllm/config/vllm.py`)**:
   - Scopes V1-specific async scheduling incompatibilities (`v1_async_sched_incompat`: non-V1 spec methods, `disable_padded_drafter_batch=True`, `pipeline_parallel_size > 1`) strictly under `if not self.use_v2_model_runner:`, preserving `async_scheduling=True` under `MRV2` while retaining the global hard-fail on ROCm `deepep_high_throughput` DBO.

## 2. Deterministic Offline Verification (`own-offline`)

Run `python experiments/flashinfer_cake_rmsnorm_sglang_ep_vllm_mrv2/run.py --check`:
- **Cake RMSNorm Grad**: `7/11` cases routed to cooperative `cp.async.bulk` ring, R200 speedup `1.2161x`, deterministic `dweight` replay drift `0.0`.
- **SGLang `gfx1250` W4A8 EP**: `expert_map_len=17`, OOB gathers reduced `72 -> 0`, EP AllReduce max error `0.0`.
- **vLLM `MRV2` Async Scheduling**: `3/3` `MRV2` speculative async cases preserved, `3/3` V1 incompatibilities and `1/1` ROCm DBO case blocked.
