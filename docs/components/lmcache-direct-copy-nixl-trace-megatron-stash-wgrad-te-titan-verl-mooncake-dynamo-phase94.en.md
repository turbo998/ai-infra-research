# LMCache SGLang Zero-SM Direct Copy, NIXL UCX End-to-End Trace Invariants, Megatron-LM Paged-Stash Warmup Peak Reduction & GTP CUDA-Graph Wgrad Release, TransformerEngine Device Sentinels, TorchTitan FA Probe, veRL DeepEP/NVSHMEM, Mooncake Master VLOG(1) 10x Speedup & Dynamo Dotted Tool Names (Phase 94)

> **中文版**: [lmcache-direct-copy-nixl-trace-megatron-stash-wgrad-te-titan-verl-mooncake-dynamo-phase94.md](lmcache-direct-copy-nixl-trace-megatron-stash-wgrad-te-titan-verl-mooncake-dynamo-phase94.md)  
> **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`  
> **Experiment Path**: `experiments/lmcache_direct_copy_nixl_trace_megatron_stash_wgrad_te_titan_verl_mooncake_dynamo_phase94/`

---

## 1. Architectural Overview

Phase 94 Track 3 covers ten upstream commits across KV/transfer infrastructure (LMCache `#6062`, NIXL `#2467`, Mooncake `#4786`, Dynamo `#16482`) and distributed training/RL stacks (Megatron-LM `#8507` & `#8503`, TransformerEngine `#3739` & `#3732`, TorchTitan `#5546`, veRL `#8354`):

1. **LMCache SGLang Component Direct Copy via `cudaMemcpyBatchAsync` (`#6062`)**: Adds `EngineKVFormat::NL_X_NB_BS_NH_HS` (`format 18`, `[NB, BS, NH, HS]`) to `resolve_block_addressing`, enabling zero-SM hardware copy-engine transfers (`~0.6 us` CPU cost per entry) without the 4-slot temporary staging buffer.
2. **NIXL UCX End-to-End Trace Context & Timestamp Verification (`#2467`)**: Asserts W3C `context=<traceparent>` propagation and `at_us == nixl.phase.timestamp_us` across `nixl::postXferReq.write`, `nixl::submit`, `nixl::wire.submitted`, `nixl::notif.sent`, and `nixl::notif.received`.
3. **Mooncake Master DFS Capacity Exhaustion `VLOG(1)` Fast-Path (`#4786`)**: Replaces `LOG(ERROR)` (`470 B` per failed key under glog mutex while holding the shard write lock) with `VLOG(1)` for `ErrorCode::NO_AVAILABLE_HANDLE` in `MasterService::AllocateReplicas`, boosting failing `PutStart` throughput `10.0x` at 1 thread (`0.35 -> 3.5 Mops/s`) and `6.54x` at 16 threads (`0.26 -> 1.7 Mops/s`).
4. **Megatron-LM Paged-Stash Warmup Peak Reduction & GTP CUDA-Graph Wgrad Release (`#8507`, `#8503`)**: Calls `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` before allocating persistent MoE `stash_buffers` (`cuda_factor=1.10`) on `'capture' -> 'captured'` transition (saving `11.0 GiB` peak in our reference model), and drops `device=wgrads[0].device` from `_coalescing_manager` in GTP `_reduce_scatter` so eager `wgrad` tensors are freed immediately after `work.wait()`.
5. **TransformerEngine Accelerator-Device Sentinels & Userbuffers Cleanup (`#3739`, `#3732`)**: Allocates `None`-sentinel placeholder tensors on the first non-CPU input tensor's CUDA device via `_pack_on_input_device`, preventing PyTorch Inductor/CUDA Graph partition splits; removes unused `torch::CustomClassHolder` base classes from Userbuffers wrappers.
6. **TorchTitan FlashAttention Wheel Probe, veRL DeepEP 1.2.1 + NVSHMEM 3.4.5 & Dynamo Dotted Tool Names (`#5546`, `#8354`, `#16482`)**: Probes `flash_attn.cute` (SM100 FA4) and `flash_attn_interface` (SM90 FA3) via `importlib.util.find_spec`, restores `deep-ep==1.2.1` + `NVSHMEM 3.4.5` in `uv.cu130.dev5`, and allows `.` in OpenAI tool function names (`my-tool.v2`).

---

## 2. Deterministic Verification Summary (`own-offline`)

| Component & PR | Metric / Invariant | Before | After | Impact |
|---|---|---:|---:|---|
| LMCache `#6062` SGLang KV Copy | SMs used by `format 18` / CPU overhead | Kernel SMs + 4-slot buf | `0` SMs / `0.6 us` | Hardware copy-engine batching |
| NIXL `#2467` UCX Trace | Verified `traceparent` + `at_us` phases | Partial unit test | `5 / 5` UCX phases | End-to-end RDMA observability |
| Mooncake `#4786` Master DFS | Failing `PutStart` throughput (1T / 16T) | `0.35` / `0.26 Mops/s` | `3.50` / `1.70 Mops/s` | `10.0x` / `6.54x` shard-lock relief |
| Megatron-LM `#8507` / `#8503` | Warmup peak HBM / GTP wgrad retention | `23.0 GiB` / retained | `12.0 GiB` / freed | `-11.0 GiB` peak & zero CG leak |
| TransformerEngine `#3739` | Custom-op `None` sentinel device | `cpu` (1 split) | `cuda:0` (0 splits) | Unbroken CUDA Graph tree |
| TorchTitan `#5546` / veRL `#8354` | SM90/SM100 FA probe / DeepEP+NVSHMEM | Import crash / missing `.so` | Safe fallback / `1.2.1+3.4.5` | Resilient backend & EP packaging |
