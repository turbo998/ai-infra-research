# FlashInfer KCP Backward Carry-Plane Reuse, QSA NVFP4 Sparse Decode Attention, MegaMHC Launch-Epochs Guard, and TileLang Vectorization/Atomic Safety Analysis

> **Bilingual Navigation / 双语导航**: [中文版](flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang.md) | [English Version](flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang.en.md)
> **Evidence IDs**: `src-flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang-2026` · `ev-flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang-2026` (`priority: 73`, Phase 64 Track A)
> **Reproducible Experiment**: `python experiments/flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang/run.py --check`

---

## 1. Architect Problem Statement & Engineering Context

Across long-context linear attention training (KDA Context-Parallel), NVFP4 sparse paged decode attention (QSA), and persistent epoch-synchronized fused MegaMHC kernels, systems architects face four fundamental performance and correctness trade-offs:

1. **Redundant Backward Segment Re-Walk in KDA Context-Parallel (`boundary_mode="saved"`)**: When a rank splits its packed document fragments into $P > 1$ segments to saturate SM waves, the forward pass already computes the per-segment FP32 carry states. Discarding that carry plane forces the backward pass to re-launch the K1 `_offset` segment re-walk kernel and carry-merge kernels.
2. **Register Dequantization and L2 Thrashing in NVFP4 Sparse Paged GQA Decode (Cake QSA)**: On H100 (`sm_90a`, $H_Q=24, H_{KV}=2, D=256, \text{top-}k=2051$), reading E2M1 KV pages and E4M3 block scales while pre-loading V early increases register live ranges and pollutes L2 across 2,051 sparse pages per token.
3. **Cross-Plan CTA Grid Mismatch and Sliced-View Corruption on MegaMHC `launch_epochs`**: MegaMHC relies on a persistent `uint64[sms]` `launch_epochs` tensor that advances one slot per active CTA. Sharing `launch_epochs` across plans that launch different CTA counts—or passing a sliced/offset tensor view—corrupts per-CTA epoch monotonicity.
4. **Silent Miscompilation in TileLang `VectorizeLoop` and Signed `int64` Atomic Corruption**: TIR loop vectorization only passes `(loop_var, extent, body)` to `TLVectorizer::Vectorize`, silently dropping non-unit loop steps (`step != 1`). Meanwhile, casting signed `int64_t` to unsigned `NT1` in `AtomicMax`/`AtomicMin` inverts negative-number comparisons and crashes `AtomicStore` on `int64`/`bf16`/`fp16`.

---

## 2. Core Source Changes & Mathematical Modeling

### 2.1 FlashInfer Cake KDA Context-Parallel Backward Carry-Plane Reuse (`#7117`, `ddc9524`)

In `flashinfer/experimental/cake_kda_train/cake_kcp.py` and `cake_kcp_plan.py`, when a rank's fragments are partitioned into $P > 1$ segments under `boundary_mode="saved"` (`SAVED_CARRY_DEFAULT = 1`):
- **Forward Saved-Set Extension**: `KcpSaved` retains the full FP32 `carry` plane produced during forward execution.
- **Read-Only Backward Binding**: The saved-mode backward binds `KcpSaved.carry` read-only to replay `fwdh_tape` and `dhu_tape`, **completely skipping the per-segment K1 `_offset` state re-walk and carry-merge launches** while remaining bitwise identical to `_kcp_test_knobs={"saved_carry": 0}`.
- **Carry Plane Memory Formula**: With $n_{\text{frag}}$ fragments on the rank, $P$ segments per fragment, incoming/outgoing crossing chains $n_{\text{in}}, n_{\text{out}}$, $H=64$ heads, and $\text{STATE\_TILE\_BYTES} = 128 \times 128 \times 4 = 65536\text{ B}$:
  $$\text{plane\_slots} = n_{\text{frag}} \cdot (P - 1) + \max(n_{\text{in}}, n_{\text{out}})$$
  $$\Delta M_{\text{carry}} = \begin{cases} \text{plane\_slots} \times H \times 65536\text{ B}, & P > 1 \\ 0, & P = 1 \end{cases}$$
  Across the GB300 (`sm_103a`, $H=64$) evaluation matrix (Q1–Q6, CP2/4/8/16), $\Delta M_{\text{carry}}$ is **12.0 MiB – 80.0 MiB per rank**, raising steady forward saved memory without increasing backward peak memory.
- **GB300 Measured Performance (`author-benchmark`)**: Across 144 configurations evaluated in two independent sessions (s1/s2), B3 (`forward + checkpoint replay + backward`) achieves **`0.9990x – 1.0372x`** speedup in s1 and **`0.9950x – 1.0387x`** in s2 (**34 / 144** rows faster by $>2\%$ in both sessions, **101 / 144** within `0.98–1.02x`, and **0 rows** slower by $>2\%$ in both). The 8 sampled high-CP contiguous/zigzag B3 rows achieve a geomean speedup of **`1.0285x` (s1) / `1.0284x` (s2)**.

### 2.2 FlashInfer Cake QSA NVFP4 Sparse Paged GQA Decode Optimization (`#7103`, `34d28b3`)

Across all four Cake QSA variants (`NHD`/`HND` $\times$ Fused cooperative / Split decode+merge) in `flashinfer/experimental/cake_qsa_nvfp4`:
1. **Load K First & Restore `evict_first`**: Schedules K page loads ahead of V and restores L2 `evict_first` cache hints on streaming KV reads to prevent 2,051 sparse pages from evicting hot query/state lines.
2. **`1-prmt` Dequantization & Aligned `int2` Output Stores**: Replaces multi-instruction bit-extraction with single-instruction `prmt.b32` byte permutation for E2M1 + E4M3 dequantization and replaces inline `st.global.v2.b32` PTX with aligned `int2` stores.
- **H100 SXM (`sm_90a`) Measured Results (`author-benchmark`)**: Across 16 shapes ($H_Q=24, H_{KV}=2, D=256, \text{top-}k=2051, \text{page\_size}=16$), overall geomean speedup is **`1.0402x`**, rising to **`1.0852x`** geomean across mid-batch sizes (`rows = 16 / 32 / 64`) and peaking at **`1.127x`** at `rows = 32` in both NHD (`36.42 us -> 32.32 us`) and HND (`38.83 us -> 34.45 us`).

### 2.3 FlashInfer Cake MegaMHC SM107 Residual Policy & `_claim_launch_epochs` Guard (`#7113`, `85e886c`)

1. **SM107 Dynamic L2 `io_policy` & 2D TMA**: `cake_mega_mhc_f24c2f3f98bfe3356398_kernel.cu` removes dead 3D TMA helpers, uses 2D TMA, adds an atomic tail flag for hidden-4096 BF16 on SM107, and selects the L2 cache policy descriptor dynamically at $\lceil N_{\text{tok}}/64 \rceil \le 64$ (`0x12f0000000000000` vs `0x1000000000000000`).
2. **`_claim_launch_epochs` Weakref Grid & Full-Storage Guard**: `mega_mhc.py` tracks live plans sharing a `launch_epochs` storage key `(data_ptr, nbytes, device_idx)` under `_EPOCH_GRIDS_LOCK`, rejecting sliced/offset views (`storage_offset != 0` or `storage.nbytes() != nbytes`) and refusing any second live plan whose CTA launch grid differs from the active claim.

### 2.4 TileLang `VectorizeLoop` Non-Unit Step Guard (`#3345`, `cb26ee0`) & CUDA Atomic Type Preservation (`#3307`, `2573d47`)

1. **`VectorizeLoop` Step Validation**: `src/transform/vectorize_loop.cc` enforces `ICHECK(!op->step.has_value() || is_one(op->step.value()))` before lowering to `TLVectorizer::Vectorize`, preventing non-unit steps (`step=2`) from being silently dropped.
2. **CUDA `AtomicMax`/`AtomicMin`/`AtomicStore` Type Preservation**: `src/tl_templates/cuda/atomic.h` preserves the signed destination type `T1` in `cuda::atomic_ref<T1, cuda::thread_scope_device>` instead of casting to unsigned `NT1`, fixing signed `int64` `AtomicMax(-5, 3) == 3` and eliminating `AtomicStore` crashes on `int64`/`bf16`/`fp16`.

---

## 3. Deterministic Verification Summary (`own-offline` + `author-benchmark`)

Running `python experiments/flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang/run.py --check` verifies:

| Subsystem / Metric | Baseline / Before Fix | Current Implementation | Architectural Takeaway |
|---|---|---|---|
| **KCP B3 Backward Latency** (`Q4, CP16, contiguous`, GB300) | `9.794 / 9.788 ms` | **`9.443 / 9.423 ms`** | **`1.0372x / 1.0387x`** peak speedup by skipping `_offset` re-walk |
| **KCP Carry Plane Footprint** ($H=64$, GB300) | `0 MiB` (recomputed in bwd) | **`12.0 – 80.0 MiB / Rank`** ($P>1$), `0 B` ($P=1$) | Trades bounded saved plane for zero-regression B3 speedup |
| **QSA NVFP4 Decode** (`rows=32`, NHD / HND, H100) | `36.42 us` / `38.83 us` | **`32.32 us` / `34.45 us`** | **`1.127x`** peak speedup; **`1.0852x`** geomean at `rows=16..64` |
| **MegaMHC `_claim_launch_epochs`** | Unchecked grid/view sharing | **Rejects mismatched grids (`64 vs 148`) & sliced views** | Prevents silent per-CTA epoch counter corruption |
| **TileLang `VectorizeLoop` & `AtomicMax`** | Drops `step=2`; `AtomicMax(-5,3) -> -5` | **Rejects `step=2`; `AtomicMax(-5,3) -> 3`** | Eliminates strided loop miscompile & signed int64 inversion |
