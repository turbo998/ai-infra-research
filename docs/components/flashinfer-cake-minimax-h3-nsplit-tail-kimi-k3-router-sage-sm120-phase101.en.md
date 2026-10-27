# FlashInfer Cake MiniMax-H3 N-Split Tail, Kimi-K3 Fused Router Arm M Co-Resident CTA Guard, SM120 Sage Sparse Attention & TileLang/CuTeDSL Invariants (Phase 101)

- **Date / Phase**: 2026-10-27 · Phase 101 (`priority: 110`)
- **Component**: `gpu-software-stack`
- **Question ID**: `flashinfer-cake-minimax-h3-nsplit-tail-kimi-k3-router-sage-sm120-phase101`
- **Evidence ID**: `claim-flashinfer-minimax-h3-tail-kimi-k3-router-sage-sm120-phase101`
- **Reproducible Experiment Suite**: [`experiments/flashinfer_minimax_h3_tail_kimi_router_phase101/`](../../experiments/flashinfer_minimax_h3_tail_kimi_router_phase101/run.py)

---

## 1. Architectural Context & Core Problem

Across Blackwell (`sm_100a` / `sm_103a` / `sm_120a`) kernels and DSL compilers, four critical engineering bottlenecks arise:

1. **Tail-Wave Quantization in Persistent Block-Scaled Out-Projection GEMMs**: In the MiniMax-H3 attention output projection (`N = 5376 = 21 x 256` with fused indexed gate and residual), shapes whose final wave occupies at most half of the co-resident 2-CTA capacity waste half the GPU waiting for a full $256 \times 256$ tile cycle.
2. **Silent Partial Route-Plan Corruption on Small Devices / MIG Slices**: In the Kimi-K3 fused MoE router (896 experts, top-16), Arm M assigns a fixed slice of 7 experts (`7b .. 7b+6`) to each of its 128 plan-owner CTAs without striding. Launching on a device or MIG slice whose persistent grid capacity is below 128 CTAs leaves high-numbered experts unwritten.
3. **Rescale-Vote Dependency Stalls in SM120 Block-Sparse Sage Attention**: On `sm_120a` (RTX PRO 6000 / RTX 5090), waiting for the cross-thread rescale vote before computing the first half of $P$ exposes pipeline latency between INT8 $QK^T$ and FP8 (E4M3) $PV$.
4. **Silent Compiler Fallbacks in TileLang and CuTeDSL**: TileLang silently dropped unsupported `round="rs"` cast annotations (such as `f32 -> E8M0`) and defaulted arch-less CUDA targets to `sm_50` on CPU-only build hosts, while CuTeDSL overwrote raw pre-pass MLIR when both `KEEP=ir` and `KEEP=ir-debug` were enabled.

---

## 2. Primary Source Empirical Analysis (10 Zero-Overlap Commits)

### 2.1 FlashInfer `#7363` (`bf82f4d797`): MiniMax-H3 BF16 & MXFP8 Out-Projection N-Split Tail Schedule

`flashinfer.diffusion_ops.minimax_h3_out_proj` (BF16) and `minimax_h3_out_proj_mxfp8` (MXFP8 W8A8, E4M3 data + UE8M0 block-32 scales) extend the NVFP4 half-width tail split to BF16 and MXFP8 on SM100 (`sm_100a`) and SM103 (`sm_103a`):

- **Eligibility Thresholds**:
  - **BF16**: $M \ge 16$ rows, partial last wave fitting in half a wave ($\le 38$ tile pairs), and total waves $\le 10$ on SM103 ($\le 9$ on SM100).
  - **MXFP8**: $M \ge 129$ rows, partial last wave fitting in half a wave, and total waves $\le 26$ on SM103 ($\le 17$ on SM100).
- **Half-Width Execution**: Eligible tail tiles execute as **$256 \times 128$ output column units**, cutting the partial last wave cost to roughly half a tile while remaining **bitwise-identical** to the whole-tile program.
- **Measured Performance (GB300 `sm_103a` / GB200 `sm_100a`, CUDA 13.4, Cold-L2 Median)**:
  - **GB300 BF16** (7 eligible shapes): **`1.012x–1.041x`** faster (geomean **`1.0343x`**, reported `1.034x`), dropping $P=8, M=4832$ from `227.4 µs` to `218.6 µs`.
  - **GB300 MXFP8** (17 eligible shapes): **`1.018x–1.057x`** faster (geomean **`1.0325x`**, reported `1.032x`), dropping $P=8, M=4825$ from `144.7 µs` to `136.9 µs` (**`1.057x`** peak).

### 2.2 FlashInfer `#7365` (`e18eb1d2ee`): Kimi-K3 Fused Router Arm M 128 Plan-Owner CTA Guard

In `flashinfer/experimental/kimi_k3_fused_router/cake_backend.py`:
- With `NUM_EXPERTS = 896` and `NUM_WARPS = 224 // 32 = 7`, the plan requires `OWNER_CTAS = 896 // 7 = 128` plan-owner CTAs.
- Unlike strided arms (`L`, `LP`, `GW`), **Arm M** (`129..256` tokens on SM100, `129..2048` on SM103) assigns experts $7b \dots 7b+6$ exclusively to CTA $b < 128$.
- `launch_grid` previously capped Arm M at `persistent_grid_cap` ($3\text{ CTAs/SM}$ on CC 10.0, $4\text{ CTAs/SM}$ on CC 10.3) without checking `cap >= 128`. With 96 CTAs, experts `672..895` (`224` experts) were left unwritten.
- PR `#7365` raises `RuntimeError` when `cap < 128`, enforcing the exact **42 vs 43 SM boundary on CC 10.0** and **31 vs 32 SM boundary on CC 10.3**.

### 2.3 FlashInfer `#7364` (`967a8ed858`), `#7405` (`4482b6b06e`), `#7465` (`46d67ef62f`)

- **SM120 Block-Sparse Sage Attention (`#7364`)**: Computes the first half of $P$ ahead of the rescale vote across all **20 metadata-specialized programs** (`bsa_attn_sm120_blk64_sage_fwd`), achieving **`1.0031x–1.0100x`** speedups on RTX PRO 6000 Blackwell (`sm_120a`, 188 SMs) with bitwise-identical BF16 outputs.
- **SM120 BF16 Non-Fused Combine Repeatability (`#7405`) & CPU-Only Relocation (`#7465`)**: Verifies bitwise repeatability of the preplanned non-fused MoE combine across eager and CUDA Graph replay after buffer poisoning, avoiding atomic-add arrival-order non-determinism.

### 2.4 CUTLASS `#3747` / `#3610` / `#3702` & TileLang `#3350` / `#2860`

- **CUTLASS 4.8.1 & CuTeDSL `MLIR_RAW` (`#3747`, `#3610`, `#3702`)**: Preserves pre-pass raw IR at `JitFunctionArtifacts.MLIR_RAW` alongside `MLIR` and retains source locations when cloning modules for the clean pass pipeline.
- **TileLang Cast & Target Validation (`#3350`, `#2860`)**: Validates `(source dtype, destination dtype, round)` before codegen dispatch so unsupported `cvt.rs` combinations fail explicitly, and rejects bare `"cuda"` targets at `_cuda_target_from_arch` to prevent TVM's silent `sm_50` fallback.

---

## 3. Deterministic Verification Summary

Running `python experiments/flashinfer_minimax_h3_tail_kimi_router_phase101/run.py --check` produces:

| Dimension | Metric | Verified Result |
|---|---|---:|
| MiniMax-H3 BF16 (GB300 `sm_103a`) | 7 tail-split shapes geomean / peak speedup | **`1.0343x`** / **`1.041x`** |
| MiniMax-H3 MXFP8 (GB300 `sm_103a`) | 17 tail-split shapes geomean / peak speedup | **`1.0325x`** / **`1.057x`** |
| Kimi-K3 Fused Router Arm M | Required co-resident owner CTAs / Min SMs (`sm_100a` / `sm_103a`) | **`128 CTAs`** / **`43 SMs` / `32 SMs`** |
| Kimi-K3 Arm M Unguarded 96-CTA Slice | Unwritten expert range / count | **`672..895`** (`224` experts unwritten) |
| SM120 Block-Sparse Sage Attention | Specialized programs / 8-shape cold-L2 speedup range | **`20` programs** / **`1.0031x–1.0100x`** |
| TileLang `cvt.rs` Cast Guard | Valid `cvt.rs` tuples accepted / Invalid tuples rejected | **`3` accepted / `3` rejected** |

---

## 4. Architect Exercises & Falsifiable Questions

1. **MIG Slice Sizing Exercise**: On a 32-SM MIG slice of B200 (`sm_100a`, 3 CTAs/SM) vs GB300 (`sm_103a`, 4 CTAs/SM), compute `persistent_grid_cap` and explain why Kimi-K3's Arm M raises `RuntimeError` on the B200 slice (`96 < 128`) while succeeding on the GB300 slice (`128 == 128`).
2. **Half-Wave Tail Amortization**: Why does MXFP8 (`minimax_h3_out_proj_mxfp8`) sustain positive tail-split speedups up to 26 waves on GB300 while BF16 caps tail-split eligibility at 10 waves?
