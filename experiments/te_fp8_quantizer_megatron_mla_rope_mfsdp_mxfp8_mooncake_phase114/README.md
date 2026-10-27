# TransformerEngine Per-Instance FP8 Quantizer, Megatron Contiguous MLA RoPE & MFSDP v2 MXFP8, TorchTitan MoE/RL, Mooncake TENT/DFS, LMCache & NIXL HF3FS (Phase 114)

Deterministic CPU/offline verification suite for Phase 114 Track C (`priority: 123`):
- `NVIDIA/TransformerEngine@14be43fb92`, `f57b8b8e61` & `NVIDIA/Megatron-LM@a4b2beebc0`, `7f8dcee979`: Per-instance `Float8CurrentScaling` `QParams` (`use_power_2_scales`), full MXFP8 C++ activation suite (`GeLU/SiLU/ReLU/QGeLU/SReLU`), contiguous MLA query RoPE loads (`50%` fewer memory transactions per warp), and MFSDP v2 MXFP8 compute + `fp8_param_gather` (`2.0x` gather compression).
- `pytorch/torchtitan@72a3a126fe`, `ab51d1311d`: Skipping unused `torch.bincount` per-expert token counts via `routing_map_TE` (`870 us/step` saved across 60 layers) and RL per-group KV cache preservation across policy updates.
- `kvcache-ai/Mooncake@129c9614de`, `c6f1d08e1c`, `LMCache/LMCache@d05401790f`, `ai-dynamo/nixl@4ef0a4d9be`: Huawei `MemFabric` (`smem_trans`) TENT transport, DFS-only replica placement (`replica_num=0, dfs_replica_num=1`), eviction-controller-owned `/cache/delete` preserving pinned keys, and lock-free `std::atomic` race fixes in NIXL's HF3FS plugin.

```bash
python experiments/te_fp8_quantizer_megatron_mla_rope_mfsdp_mxfp8_mooncake_phase114/run.py --check
```
