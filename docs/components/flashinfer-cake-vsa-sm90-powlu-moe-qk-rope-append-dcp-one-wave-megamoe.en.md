# FlashInfer SM90 Cake VSA 17-Program Dispatch, CUTLASS Fused-MoE PowLU Activation, Fused QK-RoPE Append & One-Wave DCP Decode

> **Phase 52 (`priority: 61`) · Track 1 First-Hand Source Analysis**  
> **Upstream Commits**: `flashinfer-ai/flashinfer@ddf080d8a56c` (`#6877`), `366214d4325e` (`#5743`), `51eb019768fc` (`#6888`), `49aa24fbc4b3` (`#6887`), `849e8be49ff2` (`#6122`), `90c82e1268c8` (`#6684`)  
> **Deterministic Reproduction**: `experiments/flashinfer-cake-vsa-sm90-powlu-moe-qk-rope-append-dcp-one-wave-megamoe/run.py --check`

---

## 1. Self-Contained SM90 Cake VSA 17-Program Backend (`#6877`)

`flashinfer-ai/flashinfer@ddf080d8a56c87f60a25ee9df750e373c4bcbe92` turns `cake_vsa_sm90` into a self-contained Hopper (`sm_90a`) Video Sparse Attention (VSA) backend package (`flashinfer/vsa_sparse/cake_vsa_sm90/`) compiling **17 BF16 programs** in one translation unit:

- **2 Persistent Programs**:
  - `attention`: static persistent plan;
  - `attention_queue`: dynamic tile queue.
- **15 Small Programs** (targeting `s_q <= 16,384` and low-batch regimes):
  - Direct (`4`): `k1, k3, k4, k6`
  - DSM Cluster (`6`): `k2c4, k3c3, k3c6, k4c2, k4c4, k6c2`
  - Split-KV (`2`): `k4s, k6s`
  - Split-Cluster (`3`): `k6sc3, k6sc4, k6sc8`

Across 4 benchmark suites (`137` rows total) on H100 SXM 80GB (CUDA 13.0), `119 / 137` rows are faster than baseline:
- `b20` (`20` rows): Geomean **`1.0205x`** (`1.0031x – 1.0510x`, `20/20` faster)
- `ext33` (`33` rows): Geomean **`1.0515x`** (`0.9963x – 1.2524x`, `30/33` faster)
- `real39` (`39` rows): Geomean **`1.0367x`** (`0.9854x – 1.2126x`, `36/39` faster)
- `long45` (`45` rows): Geomean **`1.0120x`** (`0.9776x – 1.0695x`, `33/45` faster)

---

## 2. CUTLASS Fused-MoE `PowLU` Activation & Zero-ABI-Change Parameter Reuse (`#5743`)

`flashinfer-ai/flashinfer@366214d4325e049199c9e79aa1eb30b0d3a867f1` adds the **PowLU** (Power Linear Unit) activation to the CUTLASS fused-MoE backend:

$$\mathrm{PowLU}(g; m) = \begin{cases} g^{\frac{m}{\sqrt{g} + 1}} \cdot \sigma(g), & g > 0 \\ g \cdot \sigma(g), & g \le 0 \end{cases}$$

$$\mathrm{out} = \min(\mathrm{PowLU}(\mathrm{gate}, m), \mathrm{limit}) \cdot \mathrm{clamp}(\mathrm{up}, -\mathrm{limit}, \mathrm{limit})$$

- **Bounded Positive Peak & FP16 Safety**: For $g \le 0$, $\mathrm{PowLU}(g) \equiv \mathrm{SiLU}(g)$; for $g > 0$, as $g \to \infty$, the exponent $\frac{m}{\sqrt{g}+1} \to 0$ so $\mathrm{PowLU}(g) \to \sigma(g) \to 1$. With `DEFAULT_POWLU_M = 2.5`, the positive branch peaks at **`~4.02`** (**`~7.02`** at $m=3.5$), well below FP16 max (`65504`).
- **Zero-ABI-Breakage Reuse**: Reuses `swiglu_alpha` for `m` and `swiglu_limit` for `limit` inside `ActivationParams`; sets `ActivationType::PowLU = 11` in C++ (`InvalidType` shifted to `12`) and `ActivationType.PowLU = 13` in Python `tllm_enums.py`.

---

## 3. Fused QK-RoPE Append (`#6888`, `#6887`), One-Wave DCP Decode (`#6122`), and SM120 MegaMoE Manifest Re-Sealing (`#6684`)

1. **`cake_fused_qk_rope_fp8_append` (`#6888`) & `cake_fused_qk_rope_append` (`#6887`)**:
   - Benchmarked across `sm_90a` (H100/H200), `sm_100a` (B200), and `sm_103a` (B300/GB300):
   - **FP8 Append (`44` rows/GPU, `220` rows total)**: All `220 / 220` rows $\ge 1.00\times$, with geomeans of **`1.0423x`** (H100), **`1.0442x`** (H200), **`1.0328x`** (B200), **`1.0347x`** (B300), and **`1.0341x`** (GB300).
   - **BF16 Append (`28` rows/GPU, `140` rows total)**: `139 / 140` rows $\ge 1.00\times$, reaching **`1.1362x`** (B200), **`1.1424x`** (B300), and **`1.1428x`** (GB300) geomean speedup on Blackwell.
2. **SM100a / SM103a Whole-Tile Static One-Wave DCP Decode (`#6122`)**:
   - Adds static one-wave programs `2..5` to `dcp_spec_bf16_balanced` (`head_dim=128`), cutting GB300 latency at `b=1, s=4096, q=4` from `11.07 us` to **`9.73 us`** (**`1.138x`**);
   - Adds program `2` to `dcp_spec_bf16_fp8_d256_balanced`, cutting GB300 latency at `b=1, ctx=32768, q=4, cp=4` from `17.02 us` to **`13.73 us`** (**`1.240x`**).
3. **SM120 MegaMoE `part002.cuh` Deduplication & 18-Seal Manifest Verification (`#6684`)**:
   - Removes the byte-identical 1,090-line `cake_sm120_megamoe_kernel_part002.cuh` (sharing `part001.cuh` across `#if MOE_INSTANCE == 0` and `#elif MOE_SHARED_EXPERT == 1`), reducing device sources from `13` to `12` while verifying all `18` manifest seals and bit-identical cubin SASS.
