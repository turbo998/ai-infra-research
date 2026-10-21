# TE JAX Triton mHC & NCCL EP Import, Megatron VMM Build, LMCache MP Hooks, Dynamo, NIXL & Mooncake

Deterministic Python 3.12 standard-library reproduction of:
- `NVIDIA/TransformerEngine@53d1a3c71a` (`#3680`) & `6ca7b5aa88` (`#3639`): JAX Triton manifold Hyper-Connection (`mHC`) kernels (`20`-iter log-space Sinkhorn, `tf32x3` precision, `SdyShardingRule`) and stateless `ncclEpImportHandle` v2 EP handle reconstruction.
- `NVIDIA/Megatron-LM@6c8f14a3ec` (`#7494`): Packaged `vmm_symm_allocator.cpp` build and `CU_MEM_HANDLE_TYPE_FABRIC` fallback when IMEX is unprovisioned.
- `LMCache/LMCache@151c6c4f41` (`#5805`) & `4159596c25` (`#5946`): Multiprocess plugin transfer hooks (`transfer_object_group`, `_make_temp_buffer`) and `lmcache.inprocess` health monitor discovery.
- `ai-dynamo/dynamo@4d31cd2f18` (`#16742`) & `7929f5370e` (`#13744`), `ai-dynamo/nixl@03d9b74648` (`#2479`), & `kvcache-ai/Mooncake@4137ac0026` (`#4756`): FPM GIL release, DGDR `v1beta2` conversion envelopes, NIXL `nixl.xfer_request_id` & `nixl.xfer_execution`, and Mooncake Ascend A5 THP `memfd` & `SharedHostMapper`.

## Run

```bash
python experiments/te-jax-mhc-triton-nccl-ep-import-megatron-vmm-lmcache-mp-dynamo-nixl-mooncake/run.py --check
```
