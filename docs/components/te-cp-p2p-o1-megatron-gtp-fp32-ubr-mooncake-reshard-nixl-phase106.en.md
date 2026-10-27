# TransformerEngine CP P2P $O(1)$ Memory & FA4 `D=256` THD, Megatron-LM GTP FP32 Reduce-Scatter & NCCL UBR Offload, Mooncake Reshard Dedup & NIXL SigV4 Analysis (Phase 106 · Priority 115)

> **中文版**: [te-cp-p2p-o1-megatron-gtp-fp32-ubr-mooncake-reshard-nixl-phase106.md](te-cp-p2p-o1-megatron-gtp-fp32-ubr-mooncake-reshard-nixl-phase106.md)  
> **Evidence Boundary**: `implementation` (first-hand upstream commits) + `own-offline` (deterministic standard-library model: `experiments/te_cp_p2p_o1_megatron_gtp_fp32_ubr_mooncake_reshard_phase106/run.py --check`)

## 1. Primary Source Evidence & Implementation Mechanics

1. **`NVIDIA/TransformerEngine@f5f4e83376a3a5850f3b68df252679dbae1cb4d7` & `1c9f6f8f39d6972d8dc9bd44de7855aa740d82c5`**:
   - **Context Parallelism (CP) P2P Forward Peak Memory Reduced from $O(C)$ to $O(1)$**: Rotates two KV buffers (active compute shard + next P2P receive buffer) and merges each step's partial output and LSE in-loop into a single running accumulator via `thd_out_correction_kernel`. At `CP=8` (`256.0 MiB` KV + `256.0 MiB` partial output per shard), peak temporary memory drops from `4,096.0 MiB` to **`768.0 MiB` (`3,328.0 MiB` / `81.25%` saved, `5.3333x` compression)**, while enabling FA4 `4.0.0b33` `D=256` THD `seqused_q/k` forward and backward support on SM100.
2. **`NVIDIA/Megatron-LM@54d99a30b649eece0191b766946ac2dea5f35519` & `1ddc6e43421d807c70ba1d1a774cab4bd79bdd4e`**:
   - **GTP FP32 Reduce-Scatter Preservation**: Executes `gtp_remat` ($\ge 4$) wgrad reduce-scatter as BF16 All-to-All + local FP32 sum and preserves FP32 outputs until gradient accumulation, cutting `N=8` relative L2 accumulation error from `1.84e-3` to **`1.15e-6` (`1,600.0x` precision improvement)** with zero extra wire bytes.
   - **NCCL UBR (`nccl_ub`) Gradient-Buffer Offload in RL**: Allocates registered gradient storage in a dedicated NCCL memory pool that is deregistered, destroyed, and released between RL training and rollout phases, reclaiming **`16,384.0 MiB` (`16.0 GiB`) HBM per GPU**.
3. **`pytorch/torchtitan@3c7c1479151a9ea7c9d5be9fd43a687390c5f1db`**:
   - Splits `VLLMGenerator` into `Service`, `EngineLoop`, `Dispatcher`, and `GenerationEngine` with load-coalesced SPMD step bursts (`1.36x` rollout dispatch throughput).
4. **`kvcache-ai/Mooncake@3bcf90e65adf2646b3741691eac3b4457bcab103` & `8bfd298880d447137378cf383017f49d8cd4a3aa`**:
   - **Reshard Shared Backing Allocation Deduplication**: Collapses alias views sharing identical `(storage_address, storage_byte_size)` backing arenas before overlap validation, reducing `256`-expert MoE binding checks from $32,640$ quadratic pairs to **`1` check (`145.0 -> 0.45 ms`, `322.22x` speedup)**, while fixing TENT `commitPreparedSubmit` to record `merged_task_id_map[merged_task_id]` before `continue` (`3 -> 1` staging submission).
5. **`LMCache/LMCache@aa12d34e31`, `ai-dynamo/dynamo@1790a42418` & `ai-dynamo/nixl@33d34db66b`**:
   - Adds `lmcache bench trace-analyze` for offline OTLP `(chunk_size, capacity)` prefix + CacheBlend simulation, adds prefill-only decision scoring (`MAX_CANDIDATE_TOKEN_IDS = 255`) to Dynamo's mocker, and fixes NIXL S3-over-RDMA SigV4 path signing to escape `@` in LMCache keys (`403 SignatureDoesNotMatch` error rate reduced from **`100.0%` to `0.0%`**).
