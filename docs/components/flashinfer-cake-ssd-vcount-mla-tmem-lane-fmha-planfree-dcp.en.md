# Deep Dive: FlashInfer Cake SSD Look-Back Virtual CTA Cap, Blackwell MLA TMEM Layout-B Lane-Quarter Score Addressing & FMHA Plan-Free Static One-Wave DCP Decode

> **Lang / 语言**: [中文](./flashinfer-cake-ssd-vcount-mla-tmem-lane-fmha-planfree-dcp.md) | **English (Current)**
>
> **Evidence Class**: `implementation` (verified against upstream commits) + `author-benchmark` (author-reported measurements on NVIDIA Rubin R200 208-SM, GB300 SM103a, and B200 SM100a; `independently_reproduced=false`) + `own-offline` (deterministic Python 3.12 standard-library model `experiments/flashinfer_cake_ssd_vcount_mla_tmem_fmha_planfree/run.py --check`).

---

## 1. Architectural Context & Core Problem Statement

Across NVIDIA Blackwell (`sm_100a` B200 with 148 SMs, `sm_103a` GB300 with 152 SMs) and Rubin (`sm_107a` R200 with 208/212/216 SMs) persistent 1-CTA-per-SM kernels, the contract between the host loader, Tensor Memory (`tcgen05` TMEM) addressing, and the on-device work planner governs both latency floors and numerical correctness:

1. **Mamba2 SSD Look-Back Second-Wave Stragglers on Sub-Binned Board Variants**: The look-back chunk-parallel prefill program (`chunkpar_*`) in FlashInfer's `SSDCombined(backend="cake")` partitions its item-major tile order into $V$ virtual CTA ranges, and each CTA in the 1-CTA-per-SM launch grid claims ranges until none remain. Previously, the Python loader unconditionally passed the capability table value $V_{\text{table}}$ (`148` on `sm_100a`, `152` on `sm_103a`, `212` on `sm_107a`). On the **208-SM NVIDIA Rubin R200 board variant**, after all 208 CTAs completed their first range, the leftover $212 - 208 = 4$ ranges forced 4 CTAs to execute a full second round while 204 SMs waited at the grid barrier, inflating whole-call latency by $1.35\times\text{–}1.64\times$.
2. **Blackwell `cta_group::2` MLA Decode TMEM Layout-B Cross-Slot Score Mis-Addressing**: In `trtllm_batch_decode_with_kv_cache_mla(backend="cake")` (`mla_bf16_native_split8_pdl`) and eight kernels of `cake_mla_varq_dcp_decode`, the score MMA uses `M128 N128 cta_group::2` with two double-buffered score slots at TMEM columns `0` and `64`. Under Layout-B, keys `64..127` of row $r$ reside in TMEM lane `64 + r` at the **exact same columns** as keys `0..63`, with each softmax warp reading its own 32-lane quarter (`((warp % 4) * 32) << 16`). The affected kernels mistakenly added a column offset `n_half * 64`, sending softmax warps 2 and 3 (`n_half = 1`) into the other phase's score slot whenever a split tile held more than 64 keys. Furthermore, unit tests previously drew inputs at scale `0.05` (`0.1` for varq), keeping softmax nearly uniform so wrong-key reads hid inside `atol=1e-2`.
3. **Device Planner Overhead in E4M3 `head_dim=128` Balanced DCP Speculative Decode**: The `dcp_spec_bf16_fp8_balanced` family of `cake_fmha` previously shipped only Program 0 (`at_or_above_grid`, without the 8-slice fold) and Program 1 (`below_grid`, device planner with the 8-slice fold). For low-batch single-wave speculative decode shapes such as `prod_b1_s8192_q4_cp4_graph`, where all chunk and reduction tickets fit within a single grid wave, running the on-device planner incurred unnecessary `mbarrier` and ticket-dispatch overhead.

---

## 2. First-Hand Source Implementation Analysis

### 2.1 Cake SSD Combined Look-Back Virtual CTA Cap: `min(table_vcount, sm_count)`

In `flashinfer-ai/flashinfer` commit [`4af20723d2`](https://github.com/flashinfer-ai/flashinfer/commit/4af20723d27e650b67dbb20b0569c316b2185ded) (PR `#6884`, 2026-10-20), `flashinfer/mamba/cake_ssd_combined.py` updates `_chunk_parallel_vcount` to cap the capability table value at the device's physical `sm_count`:

```python
_CHUNK_PARALLEL_VCOUNT_BY_ARCH = {
    (10, 0): 148,
    (10, 3): 152,
    (10, 7): 212,
}

def _chunk_parallel_vcount(
    capability: tuple[int, int], sm_count: Optional[int] = None
) -> int:
    key = (int(capability[0]), int(capability[1]))
    vcount = _CHUNK_PARALLEL_VCOUNT_BY_ARCH.get(
        key, _CHUNK_PARALLEL_VCOUNT_BY_ARCH[_CHUNK_PARALLEL_VCOUNT_DEFAULT]
    )
    if sm_count is not None:
        vcount = min(int(vcount), int(sm_count))
    return max(1, min(int(vcount), _CHUNK_PARALLEL_SEGMENT_CAP))
```

- **Determinism & CUDA Graph Replay**: Floating-point outputs depend solely on `(inputs, V)` and never on the launch grid. `agg_work` remains sized for the capability table maximum (`212 * (64 * 128 + 1)` floats on `sm_107a`), while the kernel indexes at most $V$ ranges. Because `sm_count` is invariant per device index, CUDA-graph-captured calls replay the exact bound $V$.
- **Board-Variant Behavior**: The 208-SM Rubin R200 binds $V = 208$ (one range per CTA); 212/216-SM R200 boards keep $V = 212$, 148-SM B200 keeps $V = 148$, and 152-SM GB300 keeps $V = 152$ (byte-for-byte unchanged).

### 2.2 Blackwell MLA TMEM Layout-B Lane-Quarter Score Addressing Fix

In `flashinfer-ai/flashinfer` commit [`bc45bb9878`](https://github.com/flashinfer-ai/flashinfer/commit/bc45bb9878d6153fc83baa38c2ca1a5e9b3db446) (PR `#6590`, 2026-10-20), nine CUDA kernel sources (`cake_trtllm_mla_blackwell_db015a53334184a3005e_kernel.cu` and eight `cake_mla_varq_dcp_decode_*_kernel.cu` files) replace the TMEM load address before `tcgen05.ld.sync.aligned.32x32b.x32.b32`:

```cpp
// Before (erroneous n_half * 64 column offset sent warps 2 & 3 to the other phase slot):
int s_base = taddr + (unsigned int)s_off + (unsigned int)(n_half * 64) + (unsigned int)(tmem_row_base << 16);

// After (addresses the warp's 32-lane quarter at the current slot's columns s_off):
int s_base = taddr + (unsigned int)s_off + (unsigned int)(((warp % 4) * 32) << 16);
```

- **TMEM Coordinate Mechanics**: The lower 16 bits of a TMEM address encode the column (`s_off` = `0` or `64`), while bits `16..23` (`<< 16`) encode the lane (`0..127`). In Layout-B, keys `0..63` occupy lanes `0..63` (warps 0 and 1) and keys `64..127` occupy lanes `64..127` (warps 2 and 3) at the **same column offset `s_off`**.
- **Input Scale Sensitivity**: Raising test input scale from `0.05`/`0.1` to `_INPUT_SCALE = 0.5` produces a peaked softmax distribution, exposing the pre-fix `3.2e-2` max absolute error (`12.2%` mismatched elements on `b1 h128 kv1024` split-8 where each split has 128 keys) and verifying `3.1e-4` post-fix error at identical `11.84 µs` GB300 latency.

### 2.3 `cake_fmha` E4M3 `head_dim=128` Plan-Free Static One-Wave Program (Program 2)

In `flashinfer-ai/flashinfer` commit [`fda1694904`](https://github.com/flashinfer-ai/flashinfer/commit/fda16949043745adcf03ab4b598c977c5b91c3b6) (PR `#6119`, 2026-10-20), `dcp_balanced_program` in `flashinfer/cake_dcp.py` introduces a 3-way host program selector for `dcp_spec_bf16_fp8_balanced`:

$$
n_{\max} = \left\lceil \frac{\text{max\_pages\_per\_seq} \times 64}{256} \right\rceil, \quad T_{\text{total}} = B \cdot H_{kv} \cdot (n_{\max} + 8)
$$

1. **Program 0 (`at_or_above_grid`)**: Selected when $B \cdot H_{kv} \ge \text{sm\_count}$ (omits the unreachable idle-CTA 8-slice fold body).
2. **Program 2 (`static_one_wave`)**: Selected when $B \cdot H_{kv} < \text{sm\_count}$, $n_{\max} \ge 3$, and $T_{\text{total}} \le \text{sm\_count}$. Each CTA derives its ticket directly from its block index (`ticket = CTA index`) and skips the device planner entirely while producing bitwise-identical output to Program 1.
3. **Program 1 (`below_grid`)**: Default device-planner program with the 8-slice fold for all other below-grid shapes.

---

## 3. Quantitative Benchmarks & Deterministic Verification

### 3.1 Author-Reported Benchmarks (`author-benchmark`, `independently_reproduced=false`)

1. **Rubin R200 (208 SMs, `sm_107a`) Mamba2 Cake SSD Look-Back (`4af20723d2`)**:
   - `1x8192, h=8, bf16` (512 tiles, $V: 212 \to 208$): Warm `61.39 -> 40.85 µs` (**1.503x**), Cold `71.09 -> 44.93 µs` (**1.582x**).
   - `1x32768, h=8, bf16` (2048 tiles, $V: 212 \to 208$): Warm `164.55 -> 104.94 µs` (**1.568x**), Cold `167.02 -> 107.25 µs` (**1.557x**).
   - `1x65536, h=8, bf16` (4096 tiles, $V: 212 \to 208$): Warm `301.68 -> 183.87 µs` (**1.641x**), Cold `303.55 -> 185.15 µs` (**1.639x**).
   - `2x256, h=128, bf16` (512 tiles, $V: 212 \to 208$): Warm `47.73 -> 35.50 µs` (**1.345x**), Cold `59.20 -> 39.15 µs` (**1.512x**).
2. **B200 (`sm_100a`, 148 SMs) & GB300 (`sm_103a`, 152 SMs) `cake_fmha` Program 2 (`fda1694904`)**:
   - `prod_b1_s8192_q4_cp4_graph` (`max_pages_per_seq=36`, $n_{\max}=9$, $T_{\text{total}}=8 \times (9+8)=136 \le 148$): B200 latency drops from `12.58 µs` to **`10.85 µs` (1.159x)**; GB300 latency drops from `12.58 µs` to **`11.87 µs` (1.059x)**.

### 3.2 Offline Deterministic Model (`own-offline`)

Running `python experiments/flashinfer_cake_ssd_vcount_mla_tmem_fmha_planfree/run.py --check` validates all 8 arch/SM combinations for `_chunk_parallel_vcount`, the 4-warp x 2-phase TMEM Layout-B address traces and softmax scale sensitivity (`0.05` vs `0.50`), and the 8 production DCP shapes across B200 and GB300.

---

## 4. Architect Design Guidelines & Falsifiable Follow-Ups

1. **Always Cap Persistent Virtual Work Partitions at Physical `sm_count`**: Cloud SKU variants within the same compute capability (e.g., 208-SM vs 212/216-SM Rubin R200) suffer 35%–64% tail-wave penalties whenever persistent kernels hardcode capability-wide virtual CTA counts above the board's actual SM count.
2. **Enforce Peaked Logit Variance in Attention Kernel Regression Tests**: Small input scales (`0.05`–`0.1`) flatten softmax probabilities and mask severe TMEM column mis-addressing across half of the keys; require `_INPUT_SCALE >= 0.5` so cross-slot reads fail tolerance gates immediately.
3. **Falsifiable Follow-Up Question**: Can the plan-free `static_one_wave` specialization be extended to `dcp_spec_bf16_fp8_d256_balanced` (`head_dim=256`) with the same `reduce_tickets_per_tile = 8` budget on 148-SM B200?
