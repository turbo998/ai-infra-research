# Megatron MFSDP v2 Outer-Stream Overlap & MXFP8 Dual-Plane Unshard, TE THD CP Shared-Memory Guard, NCCL P2P Profiler & Mooncake URMA (Phase 179)

Deterministic offline verification suite for Phase 179 (`priority: 188`, Track 3: `distributed-training`).

## Audited Upstream Commits
- `NVIDIA/Megatron-LM` @ `037b883c9e79` (#6973), `88312648a9dd` (#8432), `2d7ea246f936` (#8478): MFSDP v2 DP-outer gradient reduction stream overlap (`84.0%` hidden, `1.18x` step speedup), `SchedulePolicy.unshard_all_planes_in_backward` for MXFP8 recomputation, and execution-order recording for prefetch (`100.0%` order match)
- `NVIDIA/TransformerEngine` @ `e619cea17e6f` (#3698) & `e614b7a9d3c3` (#3682): Shared-memory-size-aware THD CP helper kernels (`0` SMEM overflow errors) & optimized scale swizzling kernels (`1.15x` speedup)
- `NVIDIA/nccl` @ `5c1b628eca9c` & `352fb9471288`: Honor `KernelCh` from either P2P direction & read profiling bit at batch stride (`0` dropped events)
- `kvcache-ai/Mooncake` @ `03496197d71d` (#5104) & `0efca356cbc9` (#4959): Keep URMA module alive across multiple UB contexts (`0` crashes) & strip debug symbols from wheels (`72.5%` size reduction)
- `LMCache/LMCache` @ `08779cc9b5e4` (#6043) & `ai-dynamo/nixl` @ `641e24584ced` (#2440): Remove legacy SGLang KV format discovery & allocate GPUNETIO `nixlbench` OOB ports from reserved range (`0` collisions)
