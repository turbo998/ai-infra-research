# Phase 119 Track C: Megatron-LM mHC & Pinned Offload, TE FA4 THD/Permutation, Mooncake, LMCache, Dynamo & NIXL V3

Reproducible stdlib model covering 10 primary upstream commits across `NVIDIA/Megatron-LM`, `NVIDIA/TransformerEngine`, `kvcache-ai/Mooncake`, `LMCache/LMCache`, `ai-dynamo/dynamo`, and `ai-dynamo/nixl`:
- `Megatron-LM#7521` (`8e134118ba`): mHC precision controls, mean output contract, and full activation recompute with saved autocast context.
- `Megatron-LM#7961` (`57018e45e4`): Reusable 256B-aligned `PinnedActivationBuffer` (`--fine-grained-offloading-buffer-size-gib`) eliminating power-of-two pinned allocator growth.
- `TransformerEngine#3205` (`f38a1fce7f`): FA3/FA4 `pad_between_seqs=True` THD padding mask & CP gradient/output cleanup.
- `TransformerEngine#3609` (`0838ecbe4f`): Index-scatter MoE permutation reference reducing memory from `30 GB -> 1.75 GB` (`17.14x`).
- `Mooncake#4827` (`c21a541827`): `StoredWeightManifest` logical-box byte-range resharding across `4->8` and `8->4` layouts.
- `Mooncake#4971` (`a8acb4da20`): Ascend DSMI NUMA affinity lookup via `aclrtGetLogicDevIdByUserDevId`.
- `LMCache#5672` (`b540ae9ea0`): `uintptr_t` object pointers across `KernelGroupSpec` and CUDA block KV transfer.
- `LMCache#5911` (`4f7d283a4d`): Bound MP stores by `num_engine_computed_tokens` to prevent partial hybrid/DCP under-storing (`192 -> 256` tokens).
- `dynamo#16146` (`17bffaf9c7`): GMS checkpoint streaming & pre-restore BLAKE3 integrity verification (`max(1, min(4, 16 // w))` threads/shard).
- `nixl#2323` (`349514f94e`): NIXL Telemetry V3 plugin API, `AGENT_TRACE_PHASES_DROPPED = 20`, and destructor lifetime ordering.
