# SGLang MOSS-Image SP Full-Latent RoPE & Fused QKV, DSV4 Host-Free Draft Prep, MegaMoE Budget Fix, PD Bootstrap LRU, vLLM RL Per-Stage NCCL, HiSparse OOB Guard, and TRT-LLM GVR K=2048 (Phase 119 / Priority 128)

> **Evidence Classification**: Upstream `implementation` + `author-benchmark` (10 public upstream commits dated 2026-10-28) + repository `own-offline` deterministic arithmetic/topology verification (`experiments/sglang_diffusion_sp_dspark_megamoe_moss_lru_vllm_rl_nccl_hisparse_trtllm_gvr_phase119/run.py`). This note makes no unverified live GPU hardware re-benchmark claims.

## 1. Core Architectural Problems & Context

Recent production engineering across multimodal diffusion generation, DeepSeek-V4 (`dsv4` / `DSpark`) speculative decoding, MegaMoE attention data parallelism (`attn_dp`), PD disaggregation, and RL online weight transfer uncovered five systemic bottlenecks:

1. **Premature Latent Sharding Breaking Full-Image RoPE and 3-GEMM QKV Overhead in MOSS-Image**: Sharding image latents across SP ranks before entering the DiT corrupted 2D rotary embeddings over joint image/text tokens, while separate `q, k, v` projections incurred 3 GEMM launches per joint block even in `quality="high"` mode.
2. **Double-Divided MegaMoE DP Prefill Budgets and Evicted PD Bootstrap Prefixes**: `ServerArgs` already divides `chunked_prefill_size` by `attn_dp_size`, so dividing again in `MegaMoE` halved per-rank prefill throughput at `attn_dp_size=2`; meanwhile, PD prefill schedulers refreshed `waiting_queue` prefixes in `_touch_waiting_prefixes` but omitted `bootstrap_queue`, allowing radix-cache eviction during KV handshake.
3. **Single-Source RL Weight Broadcasts and Unchecked LoRA+EP Non-Linear Placement in vLLM**: `NCCLWeightTransferEngine` previously hardcoded trainer rank 0 as the sole broadcaster, blocking pipeline-parallel (PP) trainers from broadcasting stage shards directly; additionally, `FusedMoEWithLoRA` did not reject `"round_robin"` expert placement under EP.
4. **Runtime Triton JIT Recompilations in DFlash/DSpark and `trtllm-gen` Sparse MLA Over-Reads in HiSparse**: Dynamic `BLOCK_R` sizing in `OnlineAcceptanceEstimator` triggered mid-serving Triton compilations, while `trtllm-gen` sparse MLA kernels read ~250 rows (`656 B/row`) past the end of small staged prefill KV buffers.
5. **Single-Slot $K \le 1024$ Limit in TensorRT-LLM Guided Vocabulary Restriction (GVR)**: `GvrRegClusKernel` lacked dual-slot (`HINT_SLOTS = 2`) support and block64 parallel sampling/rescue for $K=2048$.

---

## 2. Primary Upstream Implementation Analysis (10 Zero-Overlap Commits)

| # | Repository & Commit | Date | Core Implementation Path & Microarchitectural Mechanism |
|---|---|---|---|
| 1 | `sgl-project/sglang@5bb7ecc2c2` (`#46664`) | 2026-10-28 | `python/sglang/multimodal_gen/configs/pipeline_configs/moss_image.py`: Overrides `shard_latents_for_sp` to return `(latents, False)` (`did_sp_shard_latents = False`) so the DiT builds full-image RoPE before sharding joint image/text tokens across SP ranks. |
| 2 | `sgl-project/sglang@357c53885f` (`#46897`) | 2026-10-28 | `python/sglang/kernels/ops/attention/dsv4/attn_metadata_kernels.py`: Computes `extend_start_loc = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens` directly on device when `seq_lens_cpu` or `extend_seq_lens_cpu` is `None`, enabling zero-host-sync DSpark draft preparation. |
| 3 | `sgl-project/sglang@15d39bd840` (`#46267`) | 2026-10-28 | `python/sglang/srt/models/megamoe.py`: Uses `local_chunked_prefill_size = cfg.chunked_prefill_size` directly under `attn_dp_enabled`, restoring **2.0x** per-rank prefill budget (`4096 -> 8192` at `attn_dp_size=2`). |
| 4 | `sgl-project/sglang@610a74d648` (`#46641`) | 2026-10-28 | `python/sglang/multimodal_gen/runtime/models/dits/moss_image.py`: Fuses joint-block `q, k, v` into `nn.Linear(dim, 3 * dim)` with strided `_norm_rope_kernel` (`TOKEN_STRIDE = 3 * dim`), cutting QKV launches `3 -> 1` in `quality="high"` while keeping 3-GEMM bit-exact views for `exact`/`lossless`. |
| 5 | `sgl-project/sglang@54be2010d3` (`#46812`) | 2026-10-28 | `python/sglang/srt/managers/scheduler.py`: Touches `reversed(bootstrap_queue[:512])` (`SGLANG_WAITING_PREFIX_REFRESH_BOOTSTRAP_MAX=512`) before `reversed(waiting_queue[:WAITING_QUEUE_PREFIX_MATCH_MAX])` in `_touch_waiting_prefixes`. |
| 6 | `vllm-project/vllm@2b678ac739` (`#62557`) | 2026-10-28 | `vllm/distributed/weight_transfer/nccl_engine.py`: Places trainer broadcasters in `[0, rank_offset)` of `StatelessProcessGroup` (`world_size = num_broadcasters + num_workers`) and routes each update via `NCCLWeightTransferUpdateInfo.src`. |
| 7 | `vllm-project/vllm@1c787a25cc` (`#62797`) | 2026-10-28 | `vllm/lora/layers/fused_moe.py`: Raises `ValueError` in `FusedMoEWithLoRA._ep_check` if `self.use_ep` and `expert_placement_strategy != "linear"`. |
| 8 | `vllm-project/vllm@458334f00f` (`#63819`) | 2026-10-28 | `vllm/v1/spec_decode/online_acceptance.py` & `dflash.py`: Pins `BLOCK_R = max(256, next_power_of_2(max_num_reqs))` with `OnlineAcceptanceEstimator.warmup()` and pins DFlash `_prepare_dflash_inputs_kernel` `BLOCK_SIZE = 128`, eliminating runtime JIT compilations. |
| 9 | `vllm-project/vllm@ebaa812475` (`#61659`) | 2026-10-28 | `vllm/v1/attention/backends/mla/hisparse.py`: Allocates `max(num_blocks, cdiv(256, block_size))` staged prefill KV blocks (`_MIN_STAGED_PREFILL_KV_ROWS = 256`, `656 B/row`) and slices `[:num_blocks]`, preventing `trtllm-gen` ~250-row OOB reads. |
| 10 | `NVIDIA/TensorRT-LLM@9cf4530787` (`#20998`) | 2026-10-28 | `tensorrt_llm/_torch/speculative/gvr.py`: Extends `GvrRegClusKernel` to `K=2048` (`BLKC=1024`, `HINT_SLOTS = 2`) with 64-thread (`32` candidates/group) block64 sampling and parallel warp rescue. |

---

## 3. Key Formulas, Data Movement & Offline Verification

### 3.1 HiSparse Minimum Staged KV Padding & GVR $K=2048$ Dual-Slot Mapping
1. **HiSparse Minimum Staged Allocation**: For block size $B_{\text{blk}}$ and requested blocks $N_{\text{req}}$, padding to `_MIN_STAGED_PREFILL_KV_ROWS = 256` guarantees at least:
   $$N_{\text{alloc}} = \max\!\left(N_{\text{req}}, \left\lceil \frac{256}{B_{\text{blk}}} \right\rceil\right), \quad S_{\min} = 256 \times 656\text{ B} = 167,936\text{ B} \approx 164\text{ KiB}$$
2. **GVR $K=2048$ Dual-Slot Block64 Sampling**: With $\text{BLKC} = 1024$, $\text{HINT\_SLOTS} = \lfloor (2048 + 1023) / 1024 \rfloor = 2$, and 64 active threads evaluate $2048 / 64 = 32$ candidates per group.

### 3.2 Offline Verification Summary (`run.py --check`)

- `moss_sp_did_shard_latents_before_rope`: `False`
- `dsv4_extend_start_loc` / `dsv4_host_syncs_required`: `[0, 4, 11, 14]` / `0`
- `megamoe_budget_restoration_ratio`: `2.0` (`4096 -> 8192` tokens/rank at `attn_dp_size=2`)
- `moss_high_quality_gemms_per_block`: `1` (`3 -> 1` QKV GEMM, `TOKEN_STRIDE = 9216`)
- `lru_touch_first_item`: `"boot_3"` (`bootstrap_queue[:512]` refreshed before `waiting_queue`)
- `nccl_world_size` / `valid_src_ranks`: `12` / `[0, 1, 2, 3]`
- `lora_ep_round_robin_rejected`: `True`
- `online_acceptance_block_r` / `dflash_fixed_block_size`: `256` / `128`
- `hisparse_oob_prevented`: `True` (`256` rows allocated >= `250`-row hardware over-read)
- `gvr_hint_slots` / `gvr_candidates_per_group`: `2` / `32`

---

## 4. Architect Exercises & Falsifiable Questions

1. **Calculation Exercise**: Under `attn_dp_size = 4` and global `--chunked-prefill-size 32768`, how many prefill tokens per rank did MegaMoE schedule before and after `#46267`? (Answer: Before `#46267`, $32768 / 4 / 4 = 2048$ tokens/rank; after `#46267`, $32768 / 4 = 8192$ tokens/rank, a $4\times$ restoration).
2. **Falsifiable Hypothesis**: Why does `_touch_waiting_prefixes` iterate over `reversed(bootstrap_queue[:512])` before `reversed(waiting_queue[:WAITING_QUEUE_PREFIX_MATCH_MAX])`? Because Touch order in an LRU radix tree leaves the last-touched items as most-recently-used, while bounding the scan to the first 512 bootstrap entries prevents unbounded scheduler loop latency during traffic spikes.
