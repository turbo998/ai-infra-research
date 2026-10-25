# FA4 FwdConfig & SM100 hd256 Bwd Schedules, veRL THD quant_align_size & Raw Log-Prob Mismatch, Megatron GDP Prepare & Quantile Balancing, Mooncake VMM/PID/TENT RDMA, NIXL, Dynamo, TE & DeepEP V2.5

Deterministic Python 3.12 stdlib simulator verifying:
1. FlashAttention-4 immutable `FwdConfig` host-only selector (`e7a6402a9ff0`), SM120 automatic `num_splits <= 1` (`1fc0221183e7`), and SM100 `hd256` backward `CtaSchedule` (`ONE_CTA`, `TWO_CTA_PIPELINED`, `TWO_CTA_SERIAL`, `76d933102819`).
2. veRL recipe-agnostic THD `quant_align_size=16` (`b1a98e1d425e`) and raw `rollout_logprobs_mismatch_count` ULP-sensitive detection (`16ec093cc4a2`).
3. Megatron-LM fused `gdp_prefill_prepare` (`b291235082b8`) and global-batch quantile expert bias balancing (`92641a421f40`).
4. Mooncake `nvlink_intra` VMM/PID-namespace diagnostics (`b66bb2b5f650`) and TENT RDMA `notify_send_wr_id_` post-success advancement (`dae22d472e31`).
5. NIXL backend phase `traceRequestId`/`traceExecution` (`0b146a82d8ef`), Dynamo `LOCAL_FLUSH_TOKENS=64` atomic counter batching (`906e89d3a304`), TransformerEngine Triton mask-map `num_out_tokens` capacity clamp (`c9c92255465b`), and DeepEP V2.5 multi-buffer & DeepJIT architecture (`def865146e95`).
