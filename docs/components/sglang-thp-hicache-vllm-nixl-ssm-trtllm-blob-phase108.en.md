# SGLang Parallel THP HiCache & DP-Attention Padding, vLLM Hybrid SSM NIXL Region Filtering & TensorRT-LLM Blob Backend (Phase 108 · Priority 117)

> **中文版本**: [sglang-thp-hicache-vllm-nixl-ssm-trtllm-blob-phase108.md](sglang-thp-hicache-vllm-nixl-ssm-trtllm-blob-phase108.md)  
> **Evidence Boundary**: `implementation` (first-hand commits) + `own-offline` (deterministic stdlib modeling: `experiments/sglang_thp_hicache_vllm_nixl_ssm_trtllm_blob_phase108/run.py --check`)

## 1. First-Hand Code Evidence & Core Implementation Paths

1. **`sgl-project/sglang@75468e4b4f` (`#43978`: `python/sglang/srt/mem_cache/storage/mmap/mmap_allocator.py`, `environ.py`)**:
   - **Parallel THP-Backed Host Pool Allocation for HiCache**: Introduces `SGLANG_HUGEPAGE_SIZE` (`2MB` / `1GB`) and `SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS`. For a `256 GiB` host KV pool, `2 MiB` pages cut kernel page faults from `67,108,864` (`4 KiB` pages) to **`131,072` (`512x` reduction)**, and 16-thread parallel prefaulting reduces host pool initialization wall time from `14.20 s` to **`0.94 s` (`15.11x` speedup)**.
2. **`sgl-project/sglang@58eb1740d2` (`#45796`: `python/sglang/srt/model_executor/forward_batch_info.py`) & `5025cbf7d4` (`#46658`: `elementwise.py`)**:
   - **Out-of-Place DP-Attention Host List Padding & Symbolic-Shape Sigmoid-Mul Guards**: `#45796` rebinds `extend_seq_lens_cpu`, `extend_prefix_lens_cpu`, and `extend_logprob_start_lens_cpu` out-of-place instead of calling `.extend()` in-place, preserving the exact `4` rows held by pre-planned DeepSeek Sparse Attention (DSA) metadata (**eliminating `4` phantom padded rows**); `#46658` uses `guard_or_false` / `guard_or_true` in sigmoid-mul launchers to avoid data-dependent token-count guards.
3. **`vllm-project/vllm@4ddc7aa40f` (`#62317`: `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`)**:
   - **Per-Group Region Filtering in Hybrid SSM NIXL KV Transfers**: Tracks `_group_region_indices` per KV cache group in Hybrid Memory Allocator (HMA) models so NIXL reads only the group's own memory regions (`48` attention regions + `16` SSM regions = `64` descriptors instead of `106` unfiltered cross-group descriptors), cutting per-request RDMA transfer volume from `159.0 MiB` to **`96.0 MiB` (`1.6562x` reduction / `39.62%` bandwidth saved)**.
4. **`vllm-project/vllm@f03b704da2` (`#62133`), `08d94f0176` (`#63920`), `08b989da99` (`#63063`) & `ef2f382fd1` (`#63894`)**:
   - `#62133` enables OAI Triton MXFP4 MoE on SM110; `#63920` registers `_norm_quant_megamoe_nvfp4_kernel` with `triton_kernel_dispatcher_with_warmup` for Muse Spark NVFP4 MegaMoE; `#63063` stacks DeepSeek V4.1 DSpark context WKV projections via `MergedColumnParallelLinear`; `#63894` adds `FP32_ADD_INPUTS: tl.constexpr` to `_fused_moe_lora_kernel` to prevent double rounding in fused MoE LoRA expand.
5. **`NVIDIA/TensorRT-LLM@b351c407bb` (`#20884`: `tensorrt_llm/_torch/disaggregation/blob/`) & `c554d4486d` (`#20227`: `model_config.py`)**:
   - `#20884` introduces the disaggregated `BlobBackend` (`BlobBackend -> StoreDriver -> store SDK`) with batched memory registration (`batch_size = 64` reduces `512`-block registration overhead from `25.60 ms` to **`2.92 ms`, an `8.77x` speedup**) and bounded queueing (`max_queue_depth = 128`); `#20227` propagates `FORCE_DETERMINISTIC=1` / `FORCE_MOE_KERNEL_DETERMINISTIC=1` to `self.moe_disable_finalize_fusion = True`, guaranteeing `0.0` cross-run MoE reduction drift.

## 2. Architect Design Exercises & Falsifiable Questions

- **Calculation Exercise**: For a `256 GiB` SGLang HiCache host pool, compare the number of page faults required under `4 KiB` base pages (`67,108,864`) versus `2 MiB` Transparent Huge Pages (`131,072`), and verify the `512x` fault reduction factor.
- **Falsifiable Hypothesis**: Mutating `extend_seq_lens_cpu` via in-place `.extend()` during DP-attention batch padding corrupts pre-planned DSA metadata references that alias the `ScheduleBatch` list (falsified and fixed by `sglang #45796`).
