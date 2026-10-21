# FlashInfer MiniMax-H3 SM120 FP8-PV Hybrid Attention, SM110 XQA Tree/Decode, SM107a Rubin SSD & DeepGEMM Round-2 Deslop

> **First-Hand Upstream Sources** (`2026-10-21`):
> - `flashinfer-ai/flashinfer@ddba2f4b2c2b8f6852a565774bc31c2ace39a5c7` (`#6936`): *feat(diffusion): add opt-in FP8-PV mode for MiniMax-H3 SM120 NVFP4 varlen attention*
> - `flashinfer-ai/flashinfer@da1e9f4bc29629724cc4de618085938743beb65a` (`#6945`): *perf(sm110_xqa): faster tree and decode routes as a cake package*
> - `flashinfer-ai/flashinfer@7601d0bb7e92390aae5276951f0e4b7c8a2b5c14` (`#6943`): *perf(ssd): faster sm_107a look-back chunk-parallel programs*
> - `flashinfer-ai/flashinfer@597e3a67ca36e1d2f3820acc9b0e3ca6817c8d2e` (`#6927`): *feat(gemm): add DeepGEMM FP8 1D1D cake package*
> - `flashinfer-ai/flashinfer@4d79be41007c65a3a416a92cda2ae4395ebfbfbf` (`#6920`): *refactor(gemm): deslop cake_deepgemm round 2*

---

## 1. MiniMax-H3 SM120 NVFP4-QK / FP8-PV Hybrid Packed-Varlen Attention (`#6936`)

In MiniMax-H3 video diffusion (`H=24`, `D=128`, 1D RoPE across the first `64` channels, non-causal packed-varlen attention), the pure NVFP4 path (`pv_mode="fp4"`) executes both `QK^T` and `PV` on 4-bit `E2M1` Tensor Cores, yielding a relative-$L_2$ error of `0.186–0.193` against an FP32 oracle. Commit `ddba2f4b2c` (`#6936`) introduces an opt-in `pv_mode="fp8"` in `flashinfer.diffusion_ops.minimax_h3_sm120_varlen_attention_nvfp4` that keeps **`QK^T` in block-scaled NVFP4 (`m16n8k64` `e2m1` with `ue4m3` 16-element block scales)** while promoting **`PV` to FP8 (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`)**. This reduces relative-$L_2$ error to **`0.136–0.143`** (**`-26.39%`** midpoint error reduction) while remaining **`1.3814x` faster (geomean)** than `minimax_h3_sm120_varlen_attention_fp8` on RTX 5090 and **`1.1876x` faster (geomean)** on RTX PRO 6000 Blackwell Server Edition.

### 1.1 Three-Launch PDL Pipeline & Adaptive Program Selection

The hybrid route (`cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention`) compiles 5 `sm_120a` programs and chains 3 launches per call with Programmatic Dependent Launch (`cudaLaunchAttributeProgrammaticStreamSerialization = 1`):

| Stage | Program | Selection Condition (`minimax_h3_sm120_nvfp4_fp8pv_programs`) | Launch Geometry | Dynamic SMEM | Responsibility |
|---|---|---|---|---|---|
| Stage 1 | `KV_STATS` | Always selected | `grid=(ceil_div(T,128)+S, H, 1)`, `block=(128,1,1)`, `4 CTAs/SM` | `16,512 B` | Reduces per-(segment, head) $\mu_K$, centered $K-\mu_K$ absmax (`k_scale`), and per-column $|V|$ absmax (`v_scale`) |
| Stage 2a | `QUANTIZE_MB3` | `tokens <= 1024` | `grid=(min(max(1,ceil_div(T,8)),max_ctas), H, 1)`, `block=(128,1,1)`, `min_blocks=3` | `0 B` | Applies half-rotated RoPE (`D_rope=64`), writes NVFP4 `Q4/K4` + `UE4M3` `Q_sf/K_sf`, and transposes/quantizes $V$ into FP8 E4M3 `Vt8 [H, 128, padded_N]` |
| Stage 2b | `QUANTIZE` | `tokens > 1024` | Same grid, `8 CTAs/SM` | `0 B` | High-occupancy NVFP4 $Q/K$ + transposed FP8 $V^T$ quantization |
| Stage 3a | `ATTENTION_SHORT` | `max_seg_len <= 4096` | `grid=(ceil_div(T,128)+S, H, 1)`, `block=(256,1,1)` (`8 warps`) | `60,416 B` | Double-buffered `K4 + Vt8` SMEM pipeline (`Q4` resident in SMEM, `60,416 B < 99 KiB`), `m16n8k64` NVFP4 `QK^T` + online softmax + `m16n8k32` FP8 `PV` |
| Stage 3b | `ATTENTION_BASE` | `max_seg_len > 4096` | Same geometry | `60,416 B` | Long-segment (`>4096`) unrolled NVFP4-QK / FP8-PV attention kernel |

### 1.2 Exact Workspace Allocation Formula

`workspace_bytes_nvfp4_fp8pv(tokens, num_heads, num_segments)` computes:

$$\text{num\_kblocks} = \max\left(1, \left\lceil \frac{T}{128} \right\rceil\right), \quad \text{padded} = 128 \cdot \text{num\_kblocks}, \quad \text{tiles} = \left\lceil \frac{T}{128} \right\rceil + \max(1, S)$$

$$\text{Bytes} = 2 \cdot T \cdot H \cdot 64 + T \cdot H \cdot 8 + H \cdot \text{num\_kblocks} \cdot 1024 + H \cdot 128 \cdot \text{padded} + 4 \times \Big(T H + 258 S H + 257 \cdot \text{tiles} \cdot H\Big)$$

yielding **`228,440,928 B` (`217.86 MiB`)** at `T=33,472, H=24, S=1` and **`749,614,176 B` (`714.89 MiB`)** at `T=109,952, H=24, S=2`.

---

## 2. SM110 (`sm_110a` Thor) XQA Package (`#6945`) & SM107a Rubin Look-Back SSD (`#6943`)

1. **SM110 XQA `cake-build/1` Package (`#6945`)**:
   - Repackages `flashinfer/experimental/sm110_xqa` into 43 `sm_110a` programs targeting NVIDIA Jetson AGX Thor (`20` SMs, CUDA 13.4).
   - **D512 Speculative-Decode Tree Attention**: Retires `"tcgen05"`, `"register_mma"`, `"register_mma_split"`, and `"register_mma_auto"`, keeping `tmem` and `pair` `tcgen05` schedules (`kernel="auto"` selects `pair` for E4M3 KV caches when $\lceil Q_{\text{rows}} / 128 \rceil$ is even, and `tmem` otherwise). Delivers **`1.075x–1.125x`** tree speedups on Unit A (`19.552 -> 17.376 us` FP16 contiguous; `17.440 -> 16.064 us` FP8 page128).
   - **D128 Single-Token Decode**: Achieves **`1.386x–1.539x`** speedups (`17.584 / 17.824 us -> 11.616 / 11.584 us` at `B=1, cap=1024`).
2. **SM107a (NVIDIA R200 Rubin) Look-Back Chunk-Parallel Mamba2 SSD (`#6943`)**:
   - Upgrades all 6 `sm_107a` variants (`bf16`/`f16`/`f32` $\times$ `batched`/`varlen`), achieving a **`1.0839x` geomean main-kernel speedup** across 7 benchmark shapes, peaking at **`1.268x` main-kernel speedup** (`33.3 -> 26.2 us` on `cp_b4h8c16`) and **`1.196x` call-level speedup** (`92.1 -> 77.0 us` on `perf_batched_1x256`).

---

## 3. DeepGEMM FP8 1D1D Package (`#6927`) & Five-Family Round-2 Deslop (`#6920`)

1. **FP8 1D1D Cake Package (`#6927`)**: Exports `forward` (BF16 output) and `wgrad` (in-place FP32 accumulate `C += A @ B`) on `SM100/SM103` with 4 K128 `UE8M0` scales packed per `uint32` word (`66.801 us` on GB300 and `87.264 us` on GB200 at `4096x7168x4096`).
2. **Round-2 Deslop Across 5 Families (`#6920`)**:
   - Across `46` programs (`89` `SM100/SM103` builds) in 72 files, removes **`12,408` net lines (`+2,904 / -15,312`)**:
   - Folds 11 duplicate `fp4_kgroup` host bindings into 7 shared TUs and **14 kernel TUs into 4 macro-parameterized bodies** (`CAKE_KGROUP_{OUT_MODE,STORE_STAGES,LPT}`, deleting 10 `.cu` files);
   - Hoists **133 byte-identical inline `encode_tma_*_desc` functions** into **35 unique bodies** in `cake_deepgemm_tma_encoders.cuh` while preserving **100% byte-identical SASS across all 36 device-gated cubin builds**.
