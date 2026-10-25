# FlashAttention-4 SM100 MLA KV-Head Cache Key Isolation, 2-CTA dQ Register Direct Load, SM103 `setmaxnreg.dec` Trap Fix, and DeepEP/DeepGEMM/NIXL Cross-Layer Memory Ordering & Context Safety

**中文版**: [fa4-sm100-mla-kvhead-key-2cta-dq-reg-sm103-setmaxnreg-deepep-gin-nixl-obj.md](fa4-sm100-mla-kvhead-key-2cta-dq-reg-sm103-setmaxnreg-deepep-gin-nixl-obj.md)

## 1. Context and Core Architectural Problems

Across Blackwell SM100/SM103 kernels, GPU-Initiated Networking (GIN) fabrics, and NIXL multi-backend storage/network engines, five subtle silent data corruption (SDC) and hardware trap failure modes were recently uncovered and resolved upstream:

1. **FlashAttention-4 SM100 MLA Compilation Cache Collision**: Two MLA shapes with the same GQA ratio `nheads_per_kv = 64`—such as `(nheads_q=64, nheads_kv=1)` and `(nheads_q=128, nheads_kv=2)`—collided in the JIT cache when keyed only on `nheads_per_kv`.
2. **FlashAttention-4 SM100 2-CTA Backward Postprocess Shared Memory Pressure**: Staging FP32 `dQaccum` ($128 \times 128 \times 4 = 65,536\;\text{B}$) through shared memory consumed twice the SMEM footprint of the BF16 output tile `sdQ` ($32,768\;\text{B}$).
3. **FlashAttention-4 SM103 (`GB300`) `hd128`/`hd192` Decode `setmaxnreg.dec` Illegal Instruction Trap**: At `q_stage=1` with 512 threads/CTA (128-register launch ceiling), the uncapped formula computed `num_regs_other = 136 > 128`, causing `setmaxnreg.dec` to trap with an illegal instruction on GB300.
4. **DeepEP GIN Doorbell / System Fence Ordering and DeepGEMM MegaMoE Slot Reuse Race**: DeepEP's `engram_fetch.cuh` needed to ring the RDMA doorbell before `flush_async` on the final `get`; `comm.cuh` lacked `fence.acq_rel.sys` before the grid barrier when `ncclTeamTagWorld` spans both NVLink and RDMA; and DeepGEMM's `mega_moe.cuh` lacked `fence.acq_rel.cta` before arriving on `task_info_empty_barriers`.
5. **NIXL Worker-Thread Contextless VRAM Registration Failure and UCX `ERR_HANDLING_MODE_NONE` Crash**: Calling `S3AccelObjEngineImpl::registerMem` from an application worker thread with no current `CUcontext` failed in `cuMemObjGetDescriptor` (`rc=1`, `NIXL_ERR_BACKEND`), while `nixlUcxEp` unconditionally set `UCP_EP_PARAM_FIELD_ERR_HANDLER` even under `UCP_ERR_HANDLING_MODE_NONE`.

---

## 2. First-Hand Source Code Audit

### 2.1 FlashAttention-4 SM100 MLA `num_head_kv` Cache Key Isolation (`#2978`, commit `94e22c906678`)

In `flash_attn/cute/interface.py`, `_flash_attn_fwd` adds `num_head_kv if qv is not None else None` to the compilation key. Without this key:
- Compiling `(64, 1)` first and reusing it for `(128, 2)` launches tiles for only 1 KV head, leaving KV head 1 completely unwritten.
- Compiling `(128, 2)` first and reusing it for `(64, 1)` launches tiles for a phantom KV head 1 that aliases and corrupts the next token's Q row.

### 2.2 FlashAttention-4 SM100 2-CTA `dQaccum` Direct Register Load and `hd256` Deterministic Backward (`#2960`, `#2952`)

In `flash_attn/cute/flash_bwd_postprocess.py` (`47e91f1f7dd8`), when `self.use_2cta_instrs` is enabled, `dQaccum` is loaded directly from GMEM into registers following the 2-SM MMA TMEM layout (`m // 64 = cta_rank`, lanes 0–63 holding `[0, hdim/2)` and lanes 64–127 holding `[hdim/2, hdim)`), reducing the postprocess SMEM allocation from $65,536\;\text{B}$ to $32,768\;\text{B}$ (**50% reduction**). Commit `3451a2a67a24` (`#2952`) enables `deterministic=True` for `hd256` 2-CTA backward via atomic-free CTA-local reductions.

### 2.3 FlashAttention-4 SM103 `setmaxnreg.dec` Launch Register Ceiling (`#2950`, commit `037e808b7239`)

In `flash_attn/cute/flash_fwd_sm100.py`, for SM103 1-CTA causal `hd128` (`num_regs_softmax=176, num_regs_correction=64`) and 2-CTA `hd192` at `q_stage=1`:
$$\text{num\_regs\_other} = \min\left(\left\lfloor \frac{512 - 176 \times 1 - 64}{3 - 1} \right\rfloor \div 8 \times 8,\; \frac{65536}{512}\right) = \min(136, 128) = 128$$
Capping `num_regs_other` at `65536 // self.threads_per_cta` (`128`) prevents `setmaxnreg.dec` from exceeding the launch register count on GB300. Commit `1fc0221183e7` (`#2758`) adds automatic `num_splits` selection on SM120.

### 2.4 DeepEP GIN Doorbell / System Fence and DeepGEMM MegaMoE CTA Fence (`#752`, `#715`, `#441`)

- **DeepEP `engram_fetch.cuh` (`a56d6156febc`, `#752`)**: Buffers each peer's last request in `deferred_token_idx` / `deferred_src_byte_offset` so the final `get` rings the RDMA doorbell before `flush_async`.
- **DeepEP `comm.cuh` (`11d1cab604bb`, `#715`)**: Emits `ptx::fence_acq_rel_sys()` inside `gin_barrier_wo_local_sync` when `std::is_same_v<team_t, ncclTeamTagWorld>` before grid synchronization across mixed NVLink + RDMA fabrics.
- **DeepGEMM `mega_moe.cuh` (`78b69000794d`, `#441`)**: Emits `ptx::fence_acq_rel_cta()` (`fence.acq_rel.cta;`) inside `MegaMoEScheduler::release_task_info()` prior to `task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u)` so metadata reads complete before the scheduler overwrites the slot.

### 2.5 NIXL S3 Accelerated `scopedCudaDevice` and UCX Error Handler Mask (`#2380`, `#2423`)

- **NIXL `S3AccelObjEngineImpl::registerMem` (`d0319acbdb8e`, `#2380`)**: Uses RAII helper `scopedCudaDevice` to save the caller's `CUcontext` via `cuCtxGetCurrent`, select `mem.devId` via `cudaSetDevice` during `cuMemObjGetDescriptor`, and restore the caller's context (or `nullptr`) via `cuCtxSetCurrent` on scope exit.
- **NIXL `nixlUcxEp` (`c8178f563a9a`, `#2423`)**: Sets `UCP_EP_PARAM_FIELD_ERR_HANDLER` in `ep_params.field_mask` only when `err_handling_mode != UCP_ERR_HANDLING_MODE_NONE`.

---

## 3. Reproducible Verification Summary

Verified via `experiments/fa4_sm100_mla_kvhead_key_2cta_dq_reg_sm103_setmaxnreg_deepep_gin_nixl_obj/run.py`:

| Subsystem & Commit | Root Cause | Unfixed Behavior | Fixed Invariant | Evidence Class |
|---|---|---|---|---|
| FA4 SM100 MLA Cache Key (`94e22c90`) | `(64,1)` and `(128,2)` share `nheads_per_kv=64` | Head 1 unwritten or phantom head aliases next token | Includes `num_head_kv` in JIT cache key | `implementation` + `own-offline` |
| FA4 2-CTA `dQaccum` (`47e91f1f`) | FP32 `dQaccum` staged via SMEM | $65,536\;\text{B}$ SMEM footprint | Direct GMEM-to-register load ($32,768\;\text{B}$, **-50%**) | `implementation` + `own-offline` |
| FA4 SM103 `setmaxnreg` (`037e808b`) | `q_stage=1` derives `num_regs_other=136` | `136 > 128` illegal instruction trap on GB300 | `min(136, 65536 // 512) = 128` | `implementation` + `own-offline` |
| DeepEP / DeepGEMM (`a56d6156`, `11d1cab6`, `78b69000`) | Missing tail doorbell / `sys` & `cta` fences | Tail stall, weak fabric ordering, slot overwrite race | Deferred last `get` doorbell + `fence.acq_rel.{sys,cta}` | `implementation` + `own-offline` |
| NIXL S3 Accel & UCX (`d0319acb`, `c8178f56`) | Contextless worker thread / unconditional `ERR_HANDLER` | `rc=1 NIXL_ERR_BACKEND` / UCX segfault | `scopedCudaDevice` + conditional `ERR_HANDLER` bit | `implementation` + `own-offline` |
