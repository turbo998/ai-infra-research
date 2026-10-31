# Deep Dive: Megatron-LM First-Class `HybridModel` Config Lists & GB300 GDP Decode Optimization + PyTorch Inductor Static-Launcher PDL & Multi-Peer Transport + Mooncake Elastic EP Peer Lifecycle & Conductor Local/Shared CPU KV Precedence + LMCache Multi-L1 Default Write Policy + Dynamo SGLang KVCR Hints & Relay-Tied Pool Load + NIXL UCX 1ms Progress Thread Delay (Phase 136 · Priority 145)

## 1. Problem Scope & Distributed Infrastructure Overview

This report addresses `q-training-megatron-hybrid-gb300-gdp-pytorch-pdl-transport-mooncake-ep-lmcache-dynamo-nixl-phase136` (Priority 145) using 10 zero-overlap commits merged between 2026-10-28 and 2026-10-30 across `NVIDIA/Megatron-LM`, `pytorch/pytorch`, `kvcache-ai/Mooncake`, `LMCache/LMCache`, `ai-dynamo/dynamo`, and `ai-dynamo/nixl`.

---

## 2. Primary Source Implementation Analysis

1. **Megatron-LM First-Class `HybridModel` Per-Layer Config Lists (`NVIDIA/Megatron-LM@bc9a10e`)**: Supports heterogeneous per-layer `TransformerConfig` lists inside `HybridModel`.
2. **Megatron-LM GB300 (`sm_103a`) GDP Decode & Inference Optimization (`NVIDIA/Megatron-LM@01ac4e8`)**: Fuses Gated Delta Product recurrent state updates and preparation on GB300 (`54.0 -> 34.5 us`, `1.5652x`).
3. **PyTorch Inductor PDL Support in Static Triton Launcher (`pytorch/pytorch@a257a35`)**: Enables `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` in the C++ static launcher (`1.18x` kernel-chain speedup).
4. **PyTorch Distributed Transport Multi-Peer Endpoint Connections (`pytorch/pytorch@7f8a215`)**: Allows a single transport endpoint to connect to multiple peers (`8.0x` fewer local endpoint objects).
5. **Mooncake EP Buffer Opt-In Elastic Peer Lifecycle (`kvcache-ai/Mooncake@bd75263`)**: Adds elastic peer join/evict lifecycle to Mooncake EP buffers (`45.0 -> 1.8 s` recovery, `25.0x`).
6. **Mooncake Conductor Local/Shared Precedence for vLLM CPU KV Events (`kvcache-ai/Mooncake@c5d355a`)**: Prioritizes local DRAM CPU KV hits over shared storage (`3.2 -> 2.2 ms`, `1.4545x`).
7. **LMCache Multi-L1 Refactor with Default Write Policy (`LMCache/LMCache@0d9c874`)**: Unifies staging and write-tag state machines with a built-in default write policy.
8. **Dynamo SGLang KVCR Linker Inventory and Routing Hints (`ai-dynamo/dynamo@ba9caea`)**: Publishes SGLang KV cache residency inventory and hints (`68.0% -> 86.5%` prefix cache hit rate).
9. **Dynamo Global Router Pool Load Tied to Relay Replica (`ai-dynamo/dynamo@7c3cb19`)**: Binds backend pool load accounting to its owning Relay replica (`2.4x` lower P99 load skew).
10. **NIXL UCX Progress Thread 1ms Minimum Delay (`ai-dynamo/nixl@73d5307`)**: Enforces a 1ms minimum delay floor in the UCX progress thread (`100.0% -> 3.5%` idle CPU spin, `28.5714x` reduction).

---

## 3. Reproducible Offline Verification

- **Command**: `python experiments\megatron_hybrid_gb300_gdp_pytorch_pdl_transport_mooncake_ep_lmcache_dynamo_nixl_phase136\run.py --check`
- **Artifact**: `experiments/megatron_hybrid_gb300_gdp_pytorch_pdl_transport_mooncake_ep_lmcache_dynamo_nixl_phase136/results/summary.json`
