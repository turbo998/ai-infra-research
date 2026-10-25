# FA4 FwdConfig & SM100 hd256 Bwd Schedules, veRL THD Alignment & Raw Log-Prob Metrics, Megatron GDP/Quantile Balancing, Mooncake VMM/PID/TENT RDMA, NIXL, Dynamo, TE & DeepEP V2.5

> Chinese version: [fa4-fwd-config-sm100-hd256-bwd-verl-thd-logprob-megatron-gdp-qb-mooncake-nixl-dynamo-te-deepep.md](fa4-fwd-config-sm100-hd256-bwd-verl-thd-logprob-megatron-gdp-qb-mooncake-nixl-dynamo-te-deepep.md)
> Evidence grade: `implementation` (first-hand upstream commits) + `own-offline` (deterministic Python 3.12 stdlib simulator, not direct GPU/RDMA hardware re-execution)

## 1. Core Architectural Questions & Code Evolution

Phase 84 (`priority: 93`) audits 13 first-hand commits across `Dao-AILab/flash-attention`, `volcengine/verl`, `NVIDIA/Megatron-LM`, `kvcache-ai/Mooncake`, `ai-dynamo/nixl`, `ai-dynamo/dynamo`, `NVIDIA/TransformerEngine`, and `deepseek-ai/DeepEP`:

1. **FlashAttention-4 Immutable `FwdConfig`, SM120 Auto `num_splits`, and SM100 `hd256` Backward `CtaSchedule` ([#2732](https://github.com/Dao-AILab/flash-attention/commit/e7a6402a9ff099de5eb628ce207a88a3f0aa5378), [#2758](https://github.com/Dao-AILab/flash-attention/commit/1fc0221183e799363001394a12dd9e78e15f5dda), [#2964](https://github.com/Dao-AILab/flash-attention/commit/76d9331028192d06da018c034979765aa0f22899))**:
   - Introduces host-only `FwdConfig` selection/validation without reading tensors or synchronizing CUDA, caps KV stages at `32` so small `head_dim_padded=16` cannot pick 52 stages and overflow the `227 KB` SMEM limit, resolves automatic `num_splits <= 0` to `1` on SM120, and defines `CtaSchedule` (`ONE_CTA`, `TWO_CTA_PIPELINED`, `TWO_CTA_SERIAL`) for the 5 GEMMs in SM100 `hd256` backward.
2. **veRL Recipe-Agnostic THD `quant_align_size=16` & 1-ULP Raw Log-Prob Mismatch Detection ([#8325](https://github.com/verl-project/verl/commit/b1a98e1d425e44180ddbf8d87aa074dde71c5466), [#8025](https://github.com/verl-project/verl/commit/16ec093cc4a235c603c29aed1f2c4e9f551fab48))**:
   - Unifies THD sequence and top-k distillation tensor padding under `quant_align_size=16` (`min_local_rows=64`, `cp_layout="dualpipev"`), and adds `training/rollout_logprobs_mismatch_count` (summed across steps) to catch 1-ULP log-prob drift (`-0.10000000149` vs `-0.10000000894`) where `exp()` rounds to identical FP32 probabilities.
3. **Megatron-LM Fused `gdp_prefill_prepare` & Global-Batch Quantile Balancing ([#8172](https://github.com/NVIDIA/Megatron-LM/commit/b291235082b867187c6a773dcac6512785828163), [#7394](https://github.com/NVIDIA/Megatron-LM/commit/92641a421f40eeb40fc83f3f13ec83ca652c8b13))**:
   - Fuses Gated Delta Product prefill preparation into `gdp_prefill_prepare` and adds `get_updated_expert_bias_with_quantile` (`qb_histogram`, `qb_bin_bounds`, `qb_histogram_mode`).
4. **Mooncake NVLink IPC VMM/PID Diagnostics & TENT RDMA Notify Send Slot Fix ([#4873](https://github.com/kvcache-ai/Mooncake/commit/b66bb2b5f650392a2a65aa4f4dcd816808fb44e9), [#4434](https://github.com/kvcache-ai/Mooncake/commit/dae22d472e31f6aac92ee1dfe938dc1441b58850))**:
   - Diagnoses `cuMemCreate` (`expandable_segments:True`) and container PID-namespace collisions in `nvlink_intra`, and fixes `RdmaEndPoint::sendNotification()` so `notify_send_wr_id_` advances only after `ibv_post_send()` succeeds—preventing a failed post with 255 in-flight sends from wrapping to slot `0` and overwriting an active DMA buffer.
5. **NIXL Phase Tracing, Dynamo Atomic Batching, TE Triton Capacity Clamp & DeepEP V2.5 ([#2483](https://github.com/ai-dynamo/nixl/commit/0b146a82d8efb52f40b1da872bb271978f510e94), [#17067](https://github.com/ai-dynamo/dynamo/commit/906e89d3a304e1a329be716846304bb339571272), [#3347](https://github.com/NVIDIA/TransformerEngine/commit/c9c92255465b4518380561b1bbb0aebb177c0b6d), [#763](https://github.com/deepseek-ai/DeepEP/commit/def865146e95aa9ea223e277c0a8a4c6b81e5ac7))**:
   - Propagates `traceRequestId`/`traceExecution` to NIXL backend phases, batches Dynamo per-token output counters every `LOCAL_FLUSH_TOKENS=64` (**64x** fewer shared atomic adds), clamps TE Triton `_row_id_map_pass_2_kernel` destinations at `num_out_tokens`, and introduces DeepEP V2.5 shared EP/bucket/PP/Engram buffers with DeepJIT runtime compilation.

## 2. Reproducible Experiment

- Command: `python experiments/fa4_fwd_config_sm100_hd256_bwd_verl_thd_logprob_megatron_gdp_qb_mooncake_nixl_dynamo_te_deepep/run.py`
- Evidence ID: `ev-fa4-fwd-config-sm100-hd256-bwd-verl-thd-logprob-megatron-gdp-qb-mooncake-nixl-dynamo-te-deepep-2026`
