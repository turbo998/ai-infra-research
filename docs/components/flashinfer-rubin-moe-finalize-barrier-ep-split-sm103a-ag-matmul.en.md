# FlashInfer Rubin MoE Finalize Shared-Buffer Bulk-Wait Barrier, MoE EP Split Non-Local Route Exclusion, SM103a Peer-Outer Tile Order & Persistent Kimi-K3 `tcgen05.wait::ld` Drain

> **中文版**: [flashinfer-rubin-moe-finalize-barrier-ep-split-sm103a-ag-matmul.md](flashinfer-rubin-moe-finalize-barrier-ep-split-sm103a-ag-matmul.md)

- **Queue Question ID**: `flashinfer-rubin-moe-finalize-barrier-ep-split-sm103a-ag-matmul` (`GPU 软件栈`, `priority: 39`, `advanced`)
- **Evidence Class**: `implementation` (verified first-hand upstream commits) + `own-offline` (deterministic Python 3.12 standard-library verification)
- **Verified Date**: 2026-10-21 (UTC+08:00)
- **Primary Upstream Sources**:
  - `flashinfer-ai/flashinfer` commit [`91b201a670ea5d86df8e0e4d601a510f606eac01`](https://github.com/flashinfer-ai/flashinfer/commit/91b201a670ea5d86df8e0e4d601a510f606eac01) (PR `#6307`, 2026-10-20): `flashinfer/fused_moe/cute_dsl/rubin/blockscaled_contiguous_grouped_gemm_finalize_fusion.py` (blob `93bcb13737`)
  - `flashinfer-ai/flashinfer` commit [`69f082fe759f23e3e37d457da9075c11c4e15eee`](https://github.com/flashinfer-ai/flashinfer/commit/69f082fe759f23e3e37d457da9075c11c4e15eee) (PR `#6840`, 2026-10-20): `flashinfer/moe_ep/backends/split/kernel/fused_moe/{backend.py,bridge.py}` (blobs `86abac7266`, `f0a697e2a7`)
  - `flashinfer-ai/flashinfer` commit [`41b39c4af9ea7e3cad075effb97b571a57d51b78`](https://github.com/flashinfer-ai/flashinfer/commit/41b39c4af9ea7e3cad075effb97b571a57d51b78) (PR `#6841`, 2026-10-20): `flashinfer/jit/cake_all_gather_matmul.py` (blob `53f099b216`), `csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_bfloat16_ws4_k_major_sm_103a.cu` (blob `70649196ff`)
  - `flashinfer-ai/flashinfer` commit [`3bf6d3202239a107989451257c6f594cb5d2f008`](https://github.com/flashinfer-ai/flashinfer/commit/3bf6d3202239a107989451257c6f594cb5d2f008) (PR `#6740`, 2026-10-20): `flashinfer/experimental/cake_kimi_k3_attn_res/csrc/cake_kimi_k3_attn_res/cake_kimi_k3_attn_res_0d77890adbb0f7175149_kernel.cu` (blob `90fcb9011e`)

---

## 1. Architectural Problem Statement

1. **Rubin MoE Finalize Shared-Buffer `sC` Overwrite Race (`#6307`)**: Epilogue threads write per-tile outputs into shared memory `sC[sC_row, None, 0]`, issue async TMA bulk reductions, and synchronize via `self.epilog_sync_barrier.arrive_and_wait()`. Because a thread barrier does not wait for the TMA engine to finish reading `sC`, the next tile overwrites `sC` while TMA reads are still in flight.
2. **MoE EP Split Non-Local Routes Executing Wasted GEMMs (`#6840`)**: In `RANK_MAJOR` dispatch, non-local picks arrive as `-1`. Mapping them to `local_expert_offset` with weight `0.0` does **not** skip GEMMs because the unified runner filters solely by whether the global expert ID falls in `[local_expert_offset, local_expert_offset + num_local_experts)`.
3. **SM103a TP=4 Gate/Up AllGather+Matmul Tile Walk & Persistent TMEM Deallocation (`#6841`, `#6740`)**: At `ws=4, M_padded=128, N=14336`, a peer-outer (`po`, form code `4`) walk sweeps all `56` B-panels per peer once its chunk is ready. Meanwhile, in persistent Kimi-K3 kernels on SM100, a register read does not complete `tcgen05.ld`; only `tcgen05.wait::ld.sync.aligned` drains the pipe before `consumer_warp == 0` deallocates TMEM.

---

## 2. First-Hand Code Verification & Deterministic Results (`--check`)

- **Rubin Finalize Barrier (`91b201a670`)**: Inserts `cute.arch.cp_async_bulk_commit_group()` + `cute.arch.cp_async_bulk_wait_group(0, read=True)` before `self.epilog_sync_barrier.arrive_and_wait()`, reducing corrupted tiles across an 8-tile loop from `7/8` (`87.5%`) to **`0/8` (`0.0%`)**.
- **MoE EP Split Remote Sentinel (`69f082fe75`)**: Sets `masked_expert = (local_expert_offset + num_local_experts) % num_experts` when `num_experts is not None and num_local_experts < num_experts`, cutting local expert-slot evaluations across 3 CUDA Graph replays (`512` tokens, `top_k=8`) from `12,288` to **`6,821` (`-44.49%`, and `0` on all-remote replays)**.
- **SM103a `po` Twin & `tcgen05.wait::ld` (`41b39c4af9`, `3bf6d32022`)**: `po` cuts peer transitions at `[4, "bfloat16", "k_major", 128, 14336, "po"]` from `224` to **`4` (`-98.21%`)**, and `tcgen05.wait::ld.sync.aligned` before `barrier.sync 9, 288` reduces TMEM deallocation hazard warps from `9/9` to **`0/9`**.

---

## 3. Linked Evidence IDs

- `claim-flashinfer-rubin-moe-finalize-sc-bulk-wait`
- `claim-flashinfer-moe-ep-split-nonlocal-route-exclusion`
- `claim-flashinfer-sm103a-po-tile-tcgen05-wait-ld-offline`
