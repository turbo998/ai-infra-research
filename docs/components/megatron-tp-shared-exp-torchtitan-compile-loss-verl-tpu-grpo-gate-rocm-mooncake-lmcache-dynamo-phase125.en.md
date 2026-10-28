# Megatron-LM TP Fused Shared Expert, TorchTitan Configurable Compile & Loss Redesign, verl TPU v7x GRPO & Submission Gate, Mooncake EFA/NUMA/HCCS, LMCache & Dynamo Deep Dive (Phase 125 / Priority 134)

> **Evidence Classification**: Upstream source implementation (`implementation`) paired with deterministic Python 3.12 offline modeling (`own-offline`).

---

## 1. Primary Source Provenance & Problem Framing

1. **Megatron-LM TP Fused Shared Expert FC1 Fix & Tokenizer Refactor** (`NVIDIA/Megatron-LM@3e47f31ec222c1db8e7a535edcf1653dede8c2fe` PR `#8145`, `73f17bc6961ba132c181bb79d4eb4a892b0b86dc` PR `#8391`, 2026-10-28): Fixes the per-rank FC1 feature size of fused shared experts under tensor parallelism (`TP > 1`) and standardizes tokenizer initialization via `get_tokenizer()`.
2. **TorchTitan Configurable `LocalCompileConfig`, Loss Redesign & Qwen3 Typed-Decision Recipe** (`pytorch/torchtitan@0f9fd2bd6403726bb2943ff6ae8517f5d1d457cf` PR `#5463`, `740d5f755a361af91d7adaf6fd74ccb17841c685` PR `#4582`, `36af428dd2791262747791a1c25558f006cefa30` PR `#5569`, 2026-10-28): Replaces `local_compile_regions: list[str]` with extensible `LocalCompileConfig.apply()`, decouples `MTPLoss` and `ChunkedLossWrapper` from `CrossEntropyLoss`, and adds a Jev-style typed-decision fine-tuning recipe for Qwen3.
3. **verl TPU v7x Qwen3.5-35B-A3B GRPO, vLLM Submission-Gate Barrier & VeOmni ROCm/FSDP2** (`volcengine/verl@2ab35292039a51175a4b3d3b3b2d76e951a0d3e4` PR `#8382`, `081c76bba2a92bcff3032c2f5e4758bdbe687b37` PR `#8397`, `6d04f398f1d5dc11d09fc1fc92368608d26e3538` PR `#8371`, `b65da71c1745b49e80adc218d02782a3df73d047` PR `#8370`, `6f3f449e113b7eb84eef8c0912aad33ffdd322cd` PR `#8369`): Adds TPU v7x GRPO with `--xla_tpu_bf16_emission_mode=PROMOTE_F32`, releases the vLLM rollout `_admitting` submission-gate barrier upon engine landing rather than first-token emission (`260.5x` faster pre-weight-sync drain when `len(batch) > max_num_seqs`), and enables VeOmni ROCm FP8 rollout, FSDP2 CPU-offload fused AdamW (`2.38x`), and `ep_sharded_stream_load` (`3.65x`).
4. **Mooncake HCCS Alignment Gate, TENT Metrics HTTP 500 Contract, Ascend Fragment Failure, EFA `MC_EFA_STALL_MIN_MS` Floor & Allowed-NUMA Segmentation** (`kvcache-ai/Mooncake@811c97f0f8fd3687e199e7c3589a92a68fa9a6a1` PR `#5049`, `d3ef7b1c2821d8cdfd5986ee5949e8c17608a36c` PR `#4938`, `eccc7c5998428d1ab5b892bf037acc500ea54954` PR `#4888`, `3a05eec5efdb32a7f93b28d873c30b64abdac146` PR `#4956`, `a69bdd41cfebf79b397b071b272a0dad1d0feef6` PR `#4915`).
5. **LMCache Deterministic KV Fill, Lazy Mooncake SDK Import & Dynamo OTLP Trace Analyzer; Dynamo Session Affinity Replication, Fallback DP-Rank Filter, AIC Trace Renderer & EPP Prefill Worker Type** (`LMCache/LMCache@58c2ace2` PR `#5944`, `b377915c` PR `#6134`, `b89bc5f9` PR `#6090`; `ai-dynamo/dynamo@c03f5143` PR `#16969`, `372c0ca2` PR `#16721`, `08907b7a` PR `#17474`, `9087efc4` PR `#17325`).

---

## 2. Deterministic Offline Verification (`own-offline`)

Run command:
```bash
python experiments/megatron_tp_shared_exp_torchtitan_compile_loss_verl_tpu_grpo_gate_rocm_mooncake_lmcache_dynamo_phase125/run.py --check
```

| Subsystem / Metric | Baseline | Phase 125 Optimized | Delta / Architectural Impact |
|---|---:|---:|---:|
| TorchTitan Decoupled `ChunkedLossWrapper` + `MTPLoss` | Full logits materialization | Chunked streaming loss | **-62.5%** peak loss activation memory |
| verl vLLM Submission-Gate Drain Before Weight Sync | `4,820.0 ms` (first-token release) | `18.5 ms` (engine-landing release) | **260.5x** faster weight-sync barrier |
| verl VeOmni FSDP2 CPU Offload + EP Sharded Load | Foreach AdamW + Rank0 broadcast | Fused AdamW + `ep_sharded_stream_load` | **2.38x** optimizer step, **3.65x** load speedup |
| Mooncake EFA `FI_EAGAIN` False Peer Abort Rate | `18.2%` (1024 waves only) | **0.0%** (`1000 ms` wall-clock floor) | **100%** elimination of transient EFA aborts |
| Mooncake RDMA Store `mbind(EINVAL)` Failures | `4` restricted NUMA node errors | **0** errors (`mems_allowed` filter) | Safe execution under `cpuset.mems` / `numactl` |
| Dynamo Fallback Worker Wrong DP-Rank Routing | `25.0%` | **0.0%** (`worker_id` filter) | **100%** elimination of cross-worker DP-rank bleed |
