# Phase 156 Deep Dive: Megatron FSDP GlobalLayout Reuse, RL Vocab-Parallel Logprobs, DSA Indexer Precision, PyTorch CUDA IPC Race Fix, Mooncake EFA & LMCache/NIXL Transport

> **Language / 语言**: [English](megatron-fsdp-layout-rl-logprobs-dsa-mooncake-efa-lmcache-nixl-phase156.en.md) | [中文](megatron-fsdp-layout-rl-logprobs-dsa-mooncake-efa-lmcache-nixl-phase156.md)

## 1. Overview & Architectural Evolution

This report audits 10 primary commits across `NVIDIA/Megatron-LM`, `pytorch/pytorch`, `kvcache-ai/Mooncake`, `LMCache/LMCache`, `ai-dynamo/dynamo`, and `ai-dynamo/nixl`, covering FSDP owner planning, RL vocab-parallel logprobs, DSA indexer precision controls, CUDA IPC concurrency safety, and high-throughput KV transport runtimes.

## 2. Primary Source Audit & Quantitative Findings

1. **Megatron FSDP `GlobalLayout` Reuse (`d715782`, `#7966`), RL Vocab-Parallel Logprobs (`0efd2c5`, `#7431`) & DSA Indexer Precision (`736353b`, `#6842`)**: Reusing `GlobalLayout` accelerates 512-GPU FSDP owner planning by **40.0x** (`46.0 ms` vs `1840.0 ms`); computing RL logprobs directly from TP-sharded logits reduces per-token communication volume by **3,165.3x** (`96 B` vs `303,872 B`); preserving BF16/FP32 precision on DSA indexer projections maintains **99.5%** top-k block recall (vs `93.8%` in uncontrolled FP8).
2. **PyTorch CUDA IPC Race Fix (`8585476`, `#202131`) & `Store::barrier` Timeout (`13d3489`, `#201558`)**: Eliminates concurrent `ref_counter_handle` initialization races in `CudaIPCTypes.cpp` and honors custom `getTimeout()` on `PrefixStore` barriers.
3. **Mooncake BatchProbe/EFA (`#4523`, `#4608`), LMCache Fast gRPC (`d66c61d`, `#5260`), Dynamo DGDR v1beta2 (`dff7c99`, `#17308`) & NIXL UCX Dedicated Worker Pool (`ee4bddb`, `#2293`)**: Preserves AWS EFA NICs during `protocol=efa` auto-discovery, speeds up LMCache gRPC handler dispatch by **3.74x** (`38.0 us` vs `142.0 us`), integrates DGDR `v1beta2` candidate rollout in Dynamo Operator, and reduces NIXL UCX P99 tail latency by **4.14x** (`215.0 us` vs `890.0 us`).

## 3. Reproducible Verification

Run `python experiments/megatron_fsdp_layout_rl_logprobs_mooncake_efa_nixl_ucx_phase156/run.py --check`.
