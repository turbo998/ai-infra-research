# FlashAttention-4 SM100 2-CTA CLC Logical Coordinates, GQA Backward `pack_gqa`, & Per-CTA Ragged TMA Descriptors, veRL Single-Block Pinned LoRA Bucket Staging, and PyTorch DeviceMesh/XPU Hardening

> **Question ID**: `fa4-sm100-clc-packgqa-tma-verl-lora-bucket-pytorch-devicemesh-xpu` (Phase 76, `priority: 85`)  
> **Depends On**: `mooncake-memfabric-rl-affinity-lmcache-packed-mp-megatron-dcpv2-torchtitan`  
> **Evidence ID**: `ev-fa4-sm100-clc-packgqa-tma-verl-lora-pytorch-devicemesh-2026`  
> **Reproducible Experiment**: `experiments/fa4_sm100_clc_2cta_pack_gqa_ragged_tma_verl_lora_pytorch_mesh/run.py`  
> **Chinese Version**: [`fa4-sm100-clc-2cta-pack-gqa-ragged-tma-verl-lora-pytorch-mesh.md`](fa4-sm100-clc-2cta-pack-gqa-ragged-tma-verl-lora-pytorch-mesh.md)

---

## 1. Architectural Motivation & Bottlenecks

Across Blackwell (`SM100` / GB300 / B200) and Hopper (`SM90` / H100) attention kernels, RL rollout LoRA weight synchronization, and distributed/XPU runtime layers in late October 2026, six low-level correctness and performance issues were resolved:

1. **FA4 SM100 2-CTA Cluster Launch Control (CLC) Coordinate Space Mismatch**: Under `2CTA`, both CTAs of a cluster must execute the same pair-level `work_m` tile. Because the forward pass previously passed `use_cluster_idx=False` and CTA-level `num_block` to the CLC scheduler, `clc_work_to_coords` assigned `block_idx * 2 + rank` to each CTA, mixing $Q$ rows from different tiles in the 2-CTA MMA and causing a **`0.39` max error** on GB300 (`FA_CLC=1`, varlen packed GQA `hd128`).
2. **FA4 SM100 GQA Backward FP32 Atomic `dK`/`dV` Reduction vs Critical-Path Serialization**: Running one backward tile per query head required `cp.reduce.async.bulk` FP32 atomic accumulation and postprocessing for $dK$ and $dV$. Folding `qhead_per_kvhead` query heads into the $Q$ tile (`pack_gqa`) keeps $dK, dV$ in TMEM for direct store, but without a critical-path check, a single long causal sequence in a ragged MQA pack (`16/1 hd256`) serialized and regressed from `2297 us` to `3946 us` on B200.
3. **FA4 SM90/SM100 5D Wrap-Around TMA Descriptor `~128 KiB` Over-Prefetch Faults & Workspace Sizing**: The 5D wrap-around TMA descriptor used to bound ragged K/V loads had an unbounded tensor footprint, allowing the TMA engine to prefetch up to **`~128 KiB`** past out-of-bounds boxes and fault with `cudaErrorIllegalAddress` when unmapped memory followed $V$ or $K$. Furthermore, ignoring per-batch scheduler padding caused H100 (`B=49, seqlen_q=128, H=6`) to launch `582` CTAs against a `302`-slot descriptor workspace.
4. **veRL -> vLLM Bucketed IPC LoRA Sync Per-Tensor Clones & Pinned Allocations**: Synchronizing per-expert MoE LoRA adapters cloned every tensor twice on GPU and then allocated and pinned each tensor individually on CPU (`4,000–16,000` tensors per bucket).
5. **PyTorch `DeviceMesh` Unsorted Subgroup Splits, XPU `_StaticXpuLauncher` Leak, & Mooncake `msgpack-c 6.1.0` Float64 Timestamp Encoding**: `DeviceMesh` failed to sort subgroup ranks before `split_group` when `preserve_rank_order=False`; `_StaticXpuLauncher._load_kernel` used `Py_BuildValue("(OO)", ...)` which incremented the kernel capsule refcount to `2` and leaked Level Zero handles on `close()`; and Mooncake Conductor packed integral doubles as integers under `msgpack-c 6.1.0`.

---

## 2. First-Hand Upstream Commit Audit

| Repository & Commit | Date | Evidence Tier | Key Source Files & Architectural Changes |
|---|---|---|---|
| `Dao-AILab/flash-attention@98c1cee0ed59` (PR `#3034`) | `2026-10-24` | `implementation` + `author-benchmark` | `flash_attn/cute/flash_fwd_sm100.py` + `tile_scheduler.py`: unifies logical work-tile coordinates `num_work_m = ceil_div(seqlen_q, R * C * S)` with `use_cluster_idx = (cta_group_size > 1)` and `work_q_block(work_m, stage, cta_rank)`, reducing GB300 `FA_CLC=1` varlen packed GQA `hd128` max error from **`0.39` to `0.0` (bitwise equal)**. |
| `Dao-AILab/flash-attention@bc52a40aef87` (PR `#2967`) | `2026-10-23` | `implementation` + `author-benchmark` | `flash_bwd_sm100.py`, `flash_bwd_preprocess.py`, `flash_bwd_postprocess.py`, `interface.py`: packs `qhead_per_kvhead` query heads into the backward $Q$ tile (`dK`/`dV` stay in TMEM for direct store, no FP32 atomics/semaphores) with `_bwd_pack_gqa_heuristic` critical-path gating (**`+13%..+26%`** on B200 `hd128` GQA 2k, **`+4%..+11%`** on `hd256`). |
| `Dao-AILab/flash-attention@10b74d98fb26` (PR `#3071`) | `2026-10-23` | `implementation` | `flash_fwd_sm100.py`, `flash_fwd_sm90.py`, `flash_bwd_sm100.py`, `flash_bwd_sm90.py`: replaces 5D wrap-around TMA descriptors with 128-byte per-CTA GMEM descriptor updates (`TensorMapManager`) and bounds grid slots by `rows // tile + 2 * batch + 1` per head (`888` slots vs `582` launched CTAs on H100). |
| `volcengine/verl@3c6d9b43b67c` (PR `#8352`) | `2026-10-23` | `implementation` + `author-benchmark` | `verl/workers/rollout/vllm_rollout/utils.py`: stages each received IPC LoRA bucket into **one pinned CPU block** in `lora_dtype` and hands out tensor views, achieving **`4.7x` speedup (`165 -> 35 ms`)** for `4,000` tensors (`250 MiB`) and **`5.4x` speedup (`705 -> 131 ms`)** for `16,000` tensors (`500 MiB`) on GB300. |
| `pytorch/pytorch@020ab1c83528` (`#201949`), `8e9dd78a4c84` (`#200890`) & `kvcache-ai/Mooncake@12a7e9f10a41` (`#4906`) | `2026-10-25` | `implementation` | `device_mesh.py` sorts subgroup split ranks by default; `StaticXpuLauncher.cpp` uses `Py_BuildValue("(NO)", ...)` and `unique_ptr` RAII guards (`zeKernelDestroy` -> `zeModuleDestroy`); `mooncake_conductor.cpp` forces `pack_double` for `float64` timestamps. |

---

## 3. Mathematical Formulation & System Verification

### 3.1 FA4 SM100 2-CTA Logical Work-Tile Coordinate Contract
With $R = \text{m\_block\_size} = 128$, $C = \text{cta\_group\_size} \in \{1, 2\}$, and $S = \text{q\_stage} \in \{1, 2\}$, every CTA in a cluster receives the same logical $(\text{work\_m}, \text{head}, \text{batch}, \text{split})$ and maps to its $R$-row block via:
$$\text{work\_q\_block}(\text{work\_m}, \text{stage}, \text{cta\_rank}) = (\text{work\_m} \cdot S + \text{stage}) \cdot C + \text{cta\_rank}$$
with stride vector $(1, R, R \cdot C, R \cdot C \cdot S)$.

### 3.2 Per-CTA TMA Workspace Bound & veRL Single-Block Pinned LoRA Staging
- **Per-CTA TMA Workspace (`#3071`)**: Sizing the descriptor buffer by $M_{\text{blocks, bound}} = \lfloor N_{q,\text{rows}} / \text{tile\_m} \rfloor + 2B + 1$ per head yields $148 \times 6 = 888$ slots (`113,664 B`) for H100 (`B=49, S=128, H=6`), safely covering all `582` padded CTAs.
- **veRL Single-Block Pinned LoRA Staging (`#8352`)**: Allocating one contiguous pinned tensor `torch.empty(sum(sizes), dtype=lora_dtype, pin_memory=True)` per bucket turns vLLM's per-tensor `.to("cpu", dtype).pin_memory()` into a no-op identity return and cuts `16,000`-tensor (`500 MiB`) receive latency on GB300 from `705 ms` to `131 ms` (**`5.4x`**).
