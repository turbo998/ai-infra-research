# FlashAttention-4 Paged-KV CUDA Graph & SM100 HD256 SplitKV, TorchTitan GB300 4D RL Topology, Mooncake NoF & LMCache Packed Token IPC (Phase 87)

> **Evidence Classification**: First-hand Implementation Audit (`implementation` + `author-benchmark` + `own-offline` deterministic Python 3.12 stdlib model `experiments/fa4_verl_mcore_mooncake_phase87/run.py`).

## 1. Key Architectural Findings & First-Hand Source Paths

| Upstream Commit / PR | Source Paths | Architectural Change & Quantitative Metrics |
|---|---|---|
| `Dao-AILab/flash-attention#2984` (`73019affed`, 2026-10-15), `#2917` (`e9cf2c1651`, 2026-09-25), `#2891` (`7f4592e596`, 2026-09-24) | `csrc/flash_attn/flash_api.cpp`, `flash_attn/cute/{block_info,flash_fwd_sm100,sm100_hd256_2cta_fmha_forward,sm100_hd256_2cta_fmha_backward*}.py` | `#2984` guards `seqlens_k.max().item<int>()` in `mha_fwd_kvcache` with `c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None`, reducing device-to-host syncs during paged-KV CUDA graph capture from **1 to 0**; `#2917` adds `SplitKV` support and the uniform `BlockInfo.has_kv_work(n_block_min, n_block_max)` predicate across Load, MMA, Softmax, and Correction warp roles in SM100 `hd256` 2CTA forward, while fixing 2CTA cluster `total_mblocks` counting; `#2891` adds `seqused_q`/`seqused_k` support to SM100 `hd256` 2CTA backward kernels. |
| `pytorch/torchtitan#5581` (`43d1bb9801`, 2026-10-22) | `torchtitan_recipes/rl/dapo_math.py`, `tests/rl/unit_tests/cpu/test_dapo_math.py` | Reconfigures `muse_spark_816b_a42b_dapo_math_gb300_config` trainer topology on GB300 from **256 GPUs (`FSDP 64 x TP 4 x EP 4`)** to **128 GPUs (`FSDP 8 x TP 4 x CP 4 x EP 16`)** (**2.0x** fewer trainer GPUs), shrinking the routed-expert FSDP group by **8.0x** (`64 -> 8`) so each `EP=16` group stays within a 4-host GB300 NVLink domain, and mapping both global and sliding-window layers to `UlyssesVarlenInnerAttention` via `ContextParallelTransform` (`TP x CP = 16` token sharding). On 131K-context GB300 runs, peak memory is **186 GiB/GPU**, Dist-MoE scratch peaks at **22%** of the factor-16 reservation, and trainer MFU reaches **15% -> 26% -> 28%**. |
| `volcengine/verl#8284` (`d430157d15`, 2026-10-23) & `#8ced5d2f4b` (2026-10-18) | `verl/utils/kernel/{linear_cross_entropy,kernels}.py`, `verl/experimental/agent_loop/agent_loop.py` | `#8284` replaces hardcoded `"cuda"`/`"nccl"`/`torch.cuda.nvtx` in the Triton Linear Cross-Entropy host wrapper and 18 test files with `verl.utils.device` and `get_platform().nvtx_range(...)`, and adds `is_device_available()` guards to prevent CPU-only fallback crashes; `#8ced5d2f4b` validates boolean source dtypes on sampler support receipts in `agent_loop.py`. |
| `NVIDIA/Megatron-LM#6502` (`2052769557`, 2026-10-23) & `#7926` (`7d3f0fb3b5`, 2026-10-23) | `megatron/training/training.py`, `megatron/core/inference/engines/dynamic_engine.py` | `#6502` preserves checkpoint integrity settings during checkpoint conversion; `#7926` removes the deprecated `disable_ep_consensus` flag across `DynamicInferenceEngine` and inference configs. |
| `kvcache-ai/Mooncake#3424` (`0afc417657`, 2026-10-23) | `mooncake-store/include/nof/*`, `mooncake-store/src/nof/*` | Splits `SpdkWrapper` into neutral `NVMeoFInitiator` and `DmaBufferAllocator` interfaces (`MC_NOF_BACKEND=spdk|none`), isolates SPDK headers behind `mooncake_nof_spdk`, pins the SPDK environment via a refcounted `SpdkEnvGuard` to avoid `-EINVAL` on `spdk_env_init` after `spdk_env_fini`, tracks 2MB hugepages in `NofPageRegistry`, and preserves host CPU affinity across `rte_eal_init`. |
| `LMCache/LMCache#5150` (`6913259264`, 2026-10-24) & `#4410` (`00f6d71ed1`, 2026-10-24) | `lmcache/integration/vllm/lmcache_mp_metadata.py`, `lmcache/v1/kv_layer_groups.py`, `csrc/sycl/mp_mem_kernels_sycl.cpp` | `#5150` packs token IDs into a contiguous `uint32` buffer (`IPCCacheServerKey.from_token_ids`) instead of a Python tuple of `int`s, shrinking a 32,768-token key payload by **9.0x** (`1,179,696 B -> 131,072 B`); `#4410` enables engine-driven multi-group hybrid KV transfers via `KVLayerGroupsManager` with explicit padded block strides. |

## 2. Mathematical Formulation & Architect Sizing Rules

### 2.1 Expert FSDP Group Scaling in 4D RL Training
Given total trainer GPUs $N_{\text{gpu}}$, expert FSDP shard degree is:
$$\text{FSDP}_{\text{expert}} = \frac{N_{\text{gpu}}}{\text{TP} \times \text{EP}}$$
Moving from $(N_{\text{gpu}}=256, \text{TP}=4, \text{EP}=4)$ to $(N_{\text{gpu}}=128, \text{TP}=4, \text{CP}=4, \text{EP}=16)$ cuts $\text{FSDP}_{\text{expert}}$ from $64$ to $8$ (**8.0x reduction**) while increasing per-GPU sequence activation sharding $\text{TP} \times \text{CP}$ from $4$ to $16$ (**4.0x reduction** in per-GPU token activation footprint).

### 2.2 Uniform SplitKV Work Predicate on SM100 HD256
For $B_{\text{total}} = \lceil S_k / B_N \rceil$ KV blocks partitioned across $K_{\text{split}}$ splits with step $B_{\text{step}} = \lceil B_{\text{total}} / K_{\text{split}} \rceil$:
$$\operatorname{has\_kv\_work}(i) \iff i \cdot B_{\text{step}} < \min\big((i+1) \cdot B_{\text{step}}, B_{\text{total}}\big)$$
At $S_k=384, B_N=128, K_{\text{split}}=4$, splits $0, 1, 2$ execute 1 block each while split $3$ ($3 < 3$ is `False`) is uniformly skipped by all warp roles.

## 3. Reproducible Verification (`experiments/fa4_verl_mcore_mooncake_phase87`)

Running `python experiments/fa4_verl_mcore_mooncake_phase87/run.py --check` validates all deterministic metrics in `summary.json`.
