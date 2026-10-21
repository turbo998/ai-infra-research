# FlashInfer Cake Rubin (`sm_107a`) NVFP4 MLA Paged Decode and Collective MNNVL Handle Selection & Allocation Confirmation

> **中文版本**: [flashinfer-rubin-nvfp4-mla-decode-mnnvl-collective-handle.md](flashinfer-rubin-nvfp4-mla-decode-mnnvl-collective-handle.md)  
> **Question ID**: `flashinfer-rubin-nvfp4-mla-decode-mnnvl-collective-handle` (`priority: 51`, Phase 42)  
> **Offline Experiment**: [`experiments/flashinfer_rubin_nvfp4_mla_mnnvl_handle/run.py`](../../experiments/flashinfer_rubin_nvfp4_mla_mnnvl_handle/run.py) · [`summary.json`](../../experiments/flashinfer_rubin_nvfp4_mla_mnnvl_handle/summary.json)  
> **Evidence Tier**: `implementation` (primary source audit) + `author-benchmark` (author-reported R200 `sm_107a` numbers) + `own-offline` (deterministic Python stdlib model)

---

## 1. Architectural Context & Problem Statement

As NVIDIA Vera Rubin (`sm_107a`, compute capability `10.7`) bring-up progresses in open-source inference kernels, serving DeepSeek-V3/R1/V4 Multi-head Latent Attention (MLA, absorbed `LATENT = 512`, `ROPE = 64`) in the decode regime exposes two critical systems bottlenecks:

1. **NVFP4 Paged KV Cache & Multi-Token Prediction (MTP) Half-Tile Tail Imbalance**: Quantizing the 512-channel absorbed latent KV cache (`ckv`) into block-16 NVFP4 (E2M1 codes + E4M3 scales) shrinks the per-token KV footprint to **352 Bytes**. However, under speculative decoding / MTP (`q_len = 2..4`) and non-multiple-of-256 head counts (e.g., `H = 96, q_len = 4 -> 384 rows/request`), the second 256-row query tile is half padding (`128` active rows). Moreover, on R200 (`sm_107a`) with 208 or 212 SMs, packing 4-CTA clusters leaves leftover SMs that can still host 2-CTA pair clusters.
2. **MNNVL / Symmetric Memory Handle Negotiation & IMEX Fallback Deadlocks**: Across single-node 8×B200 and multi-node GB200 NVL72 deployments, `MnnvlMemory` and `SymmDeviceMemory` previously used divergent capability probes and sticky fallback flags. Single-host 8×B200 nodes with an all-zero NVML `clusterUuid` were unnecessarily barred from `CU_MEM_HANDLE_TYPE_FABRIC`, while containers on Driver 595 (where `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED == 1` without `/dev/nvidia-caps-imex-channels/channel0`) failed mid-allocation with `CUDA_ERROR_NOT_PERMITTED`.

This document analyzes two primary upstream commits merged into `flashinfer-ai/flashinfer` on `2026-10-21`:
- **Commit `adca7b87d23d426e7ae0eae6df3992d6b44532e8` (PR `#6931`)**: `flashinfer/experimental/cake_rubin_mla_nvfp4_decode` (`cake_backend.py` 1559 lines, `cake_jit.py` 763 lines).
- **Commit `a48bf8af3cdfb2da60ad5bbaefe2f9754f494c70` (PR `#6734`)**: Stateless collective MNNVL handle selection, allocation confirmation, and CFT gating in `flashinfer/comm/mnnvl.py` and `flashinfer/comm/nvlink_one_sided_alltoall.py`.

---

## 2. Cake Rubin (`sm_107a`) NVFP4 MLA Paged Decode Architecture

### 2.1 Strict `sm_107a` Gating & 352-Byte/Token NVFP4 Paged KV Layout

In `cake_jit.py:58-64` and `620-640`, the backend maps `ARCH_BY_CAPABILITY = {(10, 7): "sm_107a"}` with no family fallback. Each page holds `PAGE = 64` tokens (matching one Rubin TMA box), and each token's paged cache spans three tensors (`cake_backend.py:91-114`, `769-790`):

| Tensor | Shape `[num_pages, 64, width]` | Dtype | Bytes / Token | Role |
|---|---|---|---|---|
| `ckv_cache` | `[num_pages, 64, 256]` | `uint8` (packed E2M1) | `256 B` | 512 absorbed latent channels ($c_{KV}$), 2 E2M1 values/byte (`max=6.0`) |
| `ckv_sf_cache` | `[num_pages, 64, 32]` | `float8_e4m3fn` | `32 B` | 1 E4M3 scale per `SF_VEC = 16` latent channels (`512 / 16 = 32`) |
| `kpe_cache` | `[num_pages, 64, 64]` | `float8_e4m3fn` | `64 B` | 64 decoupled RoPE key channels $k_{PE}$ (`E4M3`, `max=448.0`) |
| **Total (`NVFP4 MLA`)** | — | — | **352 B / token** | **69.44% smaller** than BF16 MLA (`1152 B/tok`); **45.00% smaller** than FP8+BF16 RoPE (`640 B/tok`) |

### 2.2 Online BF16 $\to$ NVFP4 Query Quantizer (`CakeMlaNvfp4QueryQuantize`)

Before decode (`cake_backend.py:831-989`), `CakeMlaNvfp4QueryQuantize` converts the absorbed BF16 query `[rows, 576]` (`rows = sum(q_lens) * num_heads`) into a caller-owned `356 B/row` workspace (`q_nope` `256 B`, `q_sf` `32 B`, `q_rope` `64 B`, `q_scale` `4 B` FP32) using 1 warp per `(token, head)` row (`QUANTIZE_ROWS_PER_CTA = 4`):

1. **Unified Per-Row Scale `q_scale` (`FP32`)**:
   $$c_{\text{nope}} = \frac{1}{6 \times 448}, \quad c_{\text{rope}} = \text{kpe\_scale} \cdot \text{rcp}(448 \times \text{ckv\_scale})$$
   $$q_{\text{scale}} = \max\!\left(\max_{i \in [0,512)} |q_{\text{nope}, i}| \cdot c_{\text{nope}},\; \max_{j \in [0,64)} |q_{\text{rope}, j}| \cdot c_{\text{rope}}\right)$$
2. **Block-16 E4M3 Scales & Packed E2M1 Latent Codes**:
   $$\text{sf}_b = \text{e4m3}\!\left(\frac{\text{amax}_b}{6 \cdot q_{\text{scale}}}\right), \quad q_{\text{nope}, 16b+k} = \text{e2m1}\!\left(\frac{q_{16b+k}}{\text{sf}_b \cdot q_{\text{scale}}}\right)$$
3. **Pre-Normalized RoPE E4M3 Encoding**:
   $$q_{\text{rope}, j} = \text{e4m3}\!\left(\frac{q_{512+j} \cdot \text{kpe\_scale}}{q_{\text{scale}} \cdot \text{ckv\_scale}}\right)$$
   By absorbing $\text{kpe\_scale} / \text{ckv\_scale}$ into `q_rope` during quantization, the decode kernel reconstructs the exact attention logit with a single scalar multiplier `logit_scale = sm_scale * ckv_scale * log2(e)`:
   $$\text{logit}(q, k) = \text{sm\_scale} \cdot \text{ckv\_scale} \cdot q_{\text{scale}} \cdot \left(\langle Q_N, K_N \rangle_{\text{NVFP4}} + \langle Q_R, K_R \rangle_{\text{FP8}}\right)$$

### 2.3 Mixed 4-CTA Cluster + 2-CTA Pair Host Planner (`mla2_plan`)

Each decode work item launches `CLUSTER = 4` CTAs (`512` threads/CTA, `323,584 B` dynamic shared memory) over a `ROWS_TILE = 256` query tile and `KEY_TILE = 128` keys (2 pages). On R200 (`sm_107a`), hardware cluster placement leaves residual SMs (`UNITS_PAIRS_BY_SM_COUNT = {208: (50, 4), 212: (46, 14)}`):
- At **208 SMs**, 50 four-CTA clusters occupy 200 SMs, leaving 4 two-CTA pairs (`couples = 2`).
- At **212 SMs**, 46 four-CTA clusters occupy 184 SMs, leaving 14 two-CTA pairs (`couples = 7`).

`mla2_plan` (`cake_backend.py:134-375`) minimizes makespan + merge overhead via priority-queue list scheduling:
- **Full Tiles (`> 128` active rows) vs Half Tiles (`<= 128` active rows)**: For `H = 96, q_len = 4` (`384` rows/request), each request produces one 256-row Full Tile and one 128-row Half Tile. Half-tile splits emit **double items**, where a 4-CTA cluster executes two 2-CTA pairs in parallel over the two halves of the key range (writing partial slots `2*z` and `2*z+1`) at `0.6x` per-tile cost plus `MLA2_ITEM_COST = 7`.
- **Kernel Variant Selection (`decode` vs `decode_full`) & Adaptive Merge (`merge_w4/w2/w1`)**: Full-tile items (`n_full`) lead `item_table` (`[items, 6]` `int32`). When `n_full == items`, `select_record` dispatches `decode_full`; otherwise it dispatches `decode` (`mixed`). When `max_splits > 1`, the PDL-enabled merge stage selects `merge_w4` (`rows <= 1024`, 4 warps/row), `merge_w2` (`1025..2048`, 2 warps/row), or `merge_w1` (`> 2048`, 1 warp/row), adding `MERGE_BATCH_COST = 1.5` when splits exceed 32.
- **Decode Context Parallelism (`visible_keys`)**: Under `cp_world = W` and `cp_interleave = I`, rank $r$ owns global key indices $((k // I) \cdot W + r) \cdot I + (k \bmod I)$, and `visible_keys` (`cake_backend.py:378-390`) computes per-row local causal limits in $O(1)$.

### 2.4 Author-Reported R200 (`sm_107a`) Benchmarks (`author-benchmark`)

Across 16 configurations on pre-release NVIDIA R200 (`sm_107a`, 208 SMs, 120 MiB L2, CUDA 13.4) against TokenSpeed MLA 2 (2.0.3):
- **Overall 16-config geomean speedup**: **1.044x**.
- **8-config `Q = 4` MTP geomean speedup**: **1.104x**.
- **Peak speedup (`H96-Q4-B64-KV128K`)**: **1300.8 $\mu$s** vs **1623.8 $\mu$s** (**1.248x / 1.25x speedup**).

---

## 3. Collective MNNVL Handle Selection, Allocation Confirmation & CFT Gating

In `flashinfer/comm/mnnvl.py:622-810` (PR `#6734`, commit `a48bf8af3cdfb2da60ad5bbaefe2f9754f494c70`), `MnnvlMemory` and `SymmDeviceMemory` are unified around two stateless collective functions:

1. **`choose_mnnvl_handle_type(comm, dev_id)`**: All-gathers `(hostname, reason, cluster_uuid)` across `comm`.
   - **Single host (`len(hosts) == 1`)**: Selects `_FABRIC_HANDLE` whenever every GPU reports `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED != 0` (even if NVML `clusterUuid` is all-zero, as on single-node 8×B200), and `_POSIX_FD_HANDLE` otherwise.
   - **Multi-host (`len(hosts) > 1`)**: Requires every rank to support `_FABRIC_HANDLE` and report the exact same non-zero NVML `clusterUuid`; otherwise raises `RuntimeError` immediately with per-rank failure diagnostics and `_IMEX_HINT`.
2. **`allocate_mnnvl_handles(comm, dev_id, make_prop)`**:
   - Attempts `_create_and_import_handles(comm, prop, size)` with two collective error barriers (after `cuMemCreate`+`cuMemExportToShareableHandle`, and after `cuMemImportFromShareableHandle`).
   - If `FABRIC` creation or import fails on a single host (e.g., Driver 595 reporting `FABRIC=1` without `/dev/nvidia-caps-imex-channels/channel0`), every rank releases partial handles/FDs, logs a one-time warning with `_IMEX_HINT`, and retries collectively with `_POSIX_FD_HANDLE` (`AF_UNIX` `SCM_RIGHTS`).
3. **CFT Gating on Actual Handle Type**: `MnnvlMemory.uses_fabric_handles()` and `NVLinkOneSidedAlltoAll` verify that the allocated workspace actually uses `_FABRIC_HANDLE`, automatically disabling CFT if a single-host allocation fell back to POSIX file descriptors.

---

## 4. Deterministic Offline Verification (`own-offline`)

Running `python experiments/flashinfer_rubin_nvfp4_mla_mnnvl_handle/run.py --check` verifies:
- **352 B/token KV & 356 B/row Query Footprint**: `69.44%` KV memory reduction vs BF16 MLA (`1152 B/token`).
- **Exact `mla2_plan` Dispatch**: On 208-SM R200 (`units=50, pairs=4`), `H96-Q4-B64-KV128K` produces `items=192, n_full=64, max_splits=4`, selecting `decode` (`mixed`) + `merge_w1` and requiring `101,192,192 B` (`96.50 MiB`) of workspace; `H128-Q2-B16-KV64K` produces `items=80, n_full=80, max_splits=5`, selecting `decode_full`.
- **DCP Partitioning Bijection**: Across `cp_world=4, cp_interleave=64`, per-rank `visible_keys` sum to `bound + 1` for every query token.
- **MNNVL Collective Fallback**: Single-host 8×B200 with missing IMEX cleanly falls back from `FABRIC` to `POSIX_FD` and sets `cft_enabled = False`.
