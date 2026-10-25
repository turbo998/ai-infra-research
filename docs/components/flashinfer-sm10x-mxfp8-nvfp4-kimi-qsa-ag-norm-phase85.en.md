# FlashInfer SM100/SM103/SM107 MXFP8 Gated-Act, NVFP4 Quant+GEMM, Kimi-K3 Router & AttnRes, SM107 QSA, AG+GEMM & Norm-Combine Deep Dive (Phase 85)

> **Associated Experiment**: `experiments/flashinfer_sm10x_mxfp8_nvfp4_kimi_qsa_ag_norm_phase85/`  
> **Associated Question**: `q-flashinfer-sm10x-mxfp8-nvfp4-kimi-qsa-ag-norm-phase85` (`priority: 94`)  
> **中文版本**: [`flashinfer-sm10x-mxfp8-nvfp4-kimi-qsa-ag-norm-phase85.md`](./flashinfer-sm10x-mxfp8-nvfp4-kimi-qsa-ag-norm-phase85.md)

---

## 1. Overview & Architectural Context

Across NVIDIA Blackwell (`sm_100a` B200, 148 SMs), Blackwell Ultra (`sm_103a` B300 148 SMs / GB300 152 SMs), and Rubin target (`sm_107a` R200, 208–212 SMs), FlashInfer landed 7 major kernel optimizations addressing small-problem wave under-utilization, large-shape per-thread instruction overhead, multi-GPU All-Gather + GEMM final-pass CTA idling, and `cc=10.7` 15-bit SMEM descriptor compatibility:

1. **`cake_gated_act_mxfp8` Narrow `32x32`/`32x64` Tiles & 4-Row/Thread Programs (PR #7346, `0b352a63`)**: Optimizes `silu_and_mul_mxfp8_quantize` and `silu_and_mul_mxfp8_quantize_backward` across `sm_100a`, `sm_103a`, and `sm_107a`.
2. **`cake_all_gather_matmul` SM103a Last-Pass N-Split Twin (PR #7360, `e7f5b1e1`)**: Targets `world_size=8`, `M=1025` (`1152` padded), `N=1280, K=8192` (Llama-3.1-70B TP8 QKV).
3. **`cake_mm_fp4` 4-CTA Row-Split NVFP4 Quantizer & `N16_SPLITK3_M` GEMM (PR #7356, `941fda25`)**: Accelerates long-row per-token NVFP4 quantization and GEMM on `sm_100a` and `sm_103a`.
4. **`cake_kimi_k3_fused_router` Regenerated 28-Cell Router Schedule (PR #7355, `ee50fd01`)**: Covers all 28 `(num_tokens, block_m)` cells for Kimi-K3 (`896` experts, `top-16`).
5. **`cake_qsa_nvfp4` SM107 (`cc=10.7`) 3-Wave Persistent Walk & 15-Bit SMEM Descriptor (PR #7353, `1cfd7312`)**: Extends `decode_tcgen05_persistent` to R200 (`sm_107a`).
6. **`cake_warp_decode` 5-Deep-Ring K256 FC2 & Vectorized Top-10 Finalize (PR #7357, `ea95347a`)**: Accelerates `H=4096, E=512, top-10` routed-MoE decode.
7. **`cake_fused_norm_combine` 6-Program Dispatch & `cake_kimi_k3_attn_res` `_cta2` / `_crfnone` Persistent Programs (PR #7352 `20ede145` & PR #7338 `e9261a5e`)**.

---

## 2. Implementation Mechanics & Quantitative Formulas

### 2.1 `cake_gated_act_mxfp8`: Narrow Tiles & 4-Row/Thread Amortization (`0b352a63`)
Taking BF16 `[M, 2K]` inputs (plus `dOut [M, K]` in backward) and emitting MXFP8 E4M3 activations + `128x4` swizzled E8M0 scales `[M, K]`:
- **Small shapes**: Below the first $M \times K$ threshold, dispatches a narrow **`32x32`** CTA tile (fwd on `sm_100a`/`sm_103a`, bwd on all 3 targets) to populate all SMs.
- **GB300 (`sm_103a`) medium band**: `colwise fwd` uses a **`32x64`** tile between $2^{22} = 4,194,304$ and $2^{24} = 16,777,216$ elements.
- **Large shapes**: On large `sm_107a` shapes and the largest `sm_103a` `fwd both` shapes (`131072x8192`), dispatches a **4-rows-per-thread** kernel.
- **Measured Cold-L2 Speedups**:
  - B200 (`sm_100a`, 148 SMs): `128x128` rows improve **`1.144x–1.260x`** (`fwd colwise`: `2.67 -> 2.12 us`, **`1.260x`**).
  - GB300 (`sm_103a`, 152 SMs): `128x128 fwd colwise` `2.26 -> 2.02 us` (**`1.121x`**); `131072x8192 fwd both` `923.27 -> 904.89 us` (**`1.020x`**).
  - R200 (`sm_107a`, 208 SMs): `128x128 bwd colwise` `2.42 -> 2.17 us` (**`1.116x`**); `131072x8192 fwd rowwise` `528.30 -> 481.99 us` (**`1.096x`**).

### 2.2 `cake_all_gather_matmul`: SM103a Last-Pass N-Split Twin (`e7f5b1e1`)
For `world_size=8`, `M=1025` rows/rank (`1152` padded $\Rightarrow 9$ M-tiles of `128`), `N=1280` ($5$ N-tiles of `256`), `K=8192`:
- On the final peer pass, $P = 9 \times 5 = 45$ tiles remain. Because $2P = 90 \le \text{num\_ctas} (148)$, `main_bfloat16_ws8_k_major_cg1s2` (`192` threads, `197,632 B` dynamic SMEM, `TMEM_NCOLS=512`) splits the $P$ tiles into **$2P = 90$ half-width `128x128` tiles** across otherwise idle CTAs.
- Measured on 8x B300: Node 1 improves from `341.6 -> 336.6 us` (**`1.0116x`**), Node 2 from `340.7 -> 337.7 us` (**`1.0209x`**).

### 2.3 `cake_mm_fp4`: 4-CTA Row-Split Quantizer & `N16_SPLITK3_M` (`941fda25`)
- **4-CTA Row-Split (`_cs4`)**: Splits each long BF16 token row across a 4-CTA cluster during short-M launches, achieving up to **`1.255x`** quantizer speedup on B200 (`K=28672, M=17`: `3.601 -> 2.874 us`) and **`1.243x`** on GB300 (`K=28672, M=1`: `3.235 -> 2.604 us`).
- **`N16_SPLITK3_M`**: At `M=64, N=2048, K=7168`, an 8-token tile with `SPLITK3_CLUSTER=3` requires $(64/8) \times (2048/256) \times 3 = 192$ CTAs ($> 152$ SMs), whereas `N16_SPLITK3_M` uses a 16-token tile requiring only $96$ CTAs ($\le 148/152$ SMs). At `K=8192, N=28672, M=8192` on GB300, GEMM alone reaches **`1.144x`** (`519.5 -> 454.5 us`) and fused `quant+GEMM` reaches **`1.140x`** (`540.6 -> 474.0 us`).

### 2.4 `cake_kimi_k3_fused_router` & `cake_qsa_nvfp4` SM107 Persistent Walk (`ee50fd01`, `1cfd7312`)
- **Kimi-K3 Fused Router (`ee50fd01`)**: Regenerates all 28 `(num_tokens, block_m)` cells for `896` experts, `top-16`: GB300 (`sm_103a`) achieves **`1.10x–1.80x`** (`40.66 -> 22.61 us` at `tokens=8192, block_m=16`, **`1.80x`**; `5.89 -> 3.39 us` at `tokens=16, block_m=8`, **`1.74x`**); B200 (`sm_100a`) achieves **`1.06x–1.73x`**.
- **SM107 QSA Persistent Walk (`1cfd7312`)**: Uses a **15-bit** SMEM descriptor mask `(((smem_q_addr) >> 4) & 0x7FFF)` under `__CUDA_ARCH__ == 1070` and enforces a 3-wave bound `1 < ceil(rows * num_kv_heads * plain_splits / sm_count) <= 3` on R200 (`212` SMs). At `rows=128` (`512` work items $\Rightarrow 3$ waves) with `2` partial slots per `(row, KV head)`, R200 latency drops from `55.71 -> 46.05 us` (**`1.21x`** HND) and `54.99 -> 47.15 us` (**`1.17x`** NHD).

### 2.5 `cake_warp_decode`, `cake_fused_norm_combine` & `cake_kimi_k3_attn_res` (`ea95347a`, `20ede145`, `e9261a5e`)
- **`cake_warp_decode` (`ea95347a`)**: Expands `LinearK256Fc2` work ring depth `NUM_WORK_PIPE_STAGES: 3 -> 5`, caches `512` route-tile metadata rows in SMEM (`+8,192 B`, `SMEM_TOTAL = 187,648 B`), and vectorizes top-10 finalize to `4` features/thread (`1.113x` on B200 and `1.104x` on GB300 at `M=4`).
- **`cake_fused_norm_combine` (`20ede145`)**: Expands 8-peer fused RMSNorm + collective combine (`H=2560`) from 4 to **6 programs** (dedicated 1-token build + `3 CTAs/SM` pipelined band for `593..1772` tokens), yielding **`1.088x`** on B200 (`8.736 -> 8.032 us`) and **`1.092x`** on B300 (`9.153 -> 8.384 us`) at `tokens=1`.
- **`cake_kimi_k3_attn_res` (`e9261a5e`)**: Adds 2-CTA-per-SM (`_cta2`) and fence-free (`_crfnone`) persistent programs (`181` programs / `211` libraries), reaching **`1.1297x`** on B200, **`1.1219x`** on B300, and **`1.1097x`** on GB300.

---

## 3. Architect Design Exercises & Falsifiable Questions

1. **Sizing Exercise (AG+GEMM Last-Pass N-Split)**: Verify that at `M_padded = 1152, N = 1280`, $P = 45$ tiles and $2P = 90 \le 148$ CTAs. If `N = 2560` ($P = 90$), why must N-Split stay disabled on a 148-SM GPU? (Hint: $2P = 180 > 148$ spills into a second wave and doubles activation reload).
2. **Falsifiable Hypothesis**: On `sm_107a` (`__CUDA_ARCH__ == 1070`), masking the UMMA SMEM descriptor with `0x3FFF` instead of `0x7FFF` truncates bit 14 whenever `smem_q_addr >= 262,144 B`.
