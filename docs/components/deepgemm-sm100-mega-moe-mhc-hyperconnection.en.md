# DeepGEMM SM100 Persistent `MegaMoE`, Asymmetric TMA Epilogue & Multi-Head Hyper-Connections (`MegaMHC`) Architecture

> **Language / 语言**: [中文 (Chinese)](deepgemm-sm100-mega-moe-mhc-hyperconnection.md) | **English**

## 1. Architectural Motivation: SM100 Shared-Memory & HBM Bottlenecks in MoE + mHC Models

Next-generation sparse MoE models augmented with Multi-Head Hyper-Connections (`mHC`, `residual [T, hc_mult=4, H=7168]`) confront two hardware bottlenecks on NVIDIA Blackwell SM100 (`232,448 Bytes = 227 KiB` shared memory per SM):
1. **Shared-Memory Pressure in Persistent `MegaMoE`**: Fusing cross-rank token dispatch pulling (`smem_dispatch_size`), L1 Gate/Up GEMM, SwiGLU, cross-warp amax reduction, L2 Down GEMM, and C/D output staging (`smem_cd`) inside one persistent kernel leaves limited shared memory for TMA pipeline stages (`num_stages`). At large prefill tiles (`block_m = 240`), a naive symmetric output staging buffer (`40 × 40`) consumes `20,480 B` of SMEM;
2. **Six-Kernel HBM Round-Trip Storm in Unfused `mHC`**: Executing pre-mix projection, Sinkhorn-Knopp doubly-stochastic iterations (`num_sinkhorn_iters`), residual stream mixing, RMSNorm, FP8 quantization, and separate scaling-factor (`UE8M0`) transpositions for routed vs. shared experts requires **`984.38 MiB`** of HBM traffic at `T=4096, H=7168`.

Grounded in first-hand source code from **`deepseek-ai/DeepGEMM` (`commit 5081f8513af57e61696de4f4f04b0e707ffb503e`)**—specifically `csrc/apis/mega_moe.hpp`, `csrc/jit_kernels/heuristics/mega_moe.hpp`, `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`, `csrc/apis/mega_mhc.hpp`, and `csrc/jit_kernels/impls/sm100_mega_mhc.hpp`—this document derives the exact tiling heuristics, SMEM budget equations, and fusion gains.

---

## 2. First-Hand Source Code Mechanics (`implementation`)

### 2.1 `csrc/jit_kernels/heuristics/mega_moe.hpp`: One-Sigma Routing Margin & Asymmetric `store_block_m_l1 / l2`

In `get_block_config_for_mega_moe` (`blob 660223eb73b08365accbdb76d8a6b7d1c61d9414`), DeepGEMM models token routing skew with a one-sigma margin:

$$E_{\text{tokens}} = \frac{T \times R \times K}{E_{\text{total}}}, \qquad N_{\text{covered}} = E_{\text{tokens}} + \sqrt{E_{\text{tokens}}}$$

Because every additional M block reloads weight tiles (~128 extra rows of cost), the heuristic picks the smallest candidate in `kCandidateBlockM = {16, 32, 64, 128, 192, 240}` covering $N_{\text{covered}}$, splitting a single `>192` block into 2 blocks when `num_experts / num_ranks < 14`.

Crucially, at `block_m = 240` under `MXFP8FP4`, DeepGEMM uses **asymmetric store blocks**:
- `store_block_m_l1 = 24` (1-byte FP8 post-SwiGLU output with halved N width `block_n / 2 = 64` and halved TMA swizzle `64B`),
- `store_block_m_l2 = 8` (2-byte BF16 L2 output with full width `block_n = 128`).

This reduces `smem_cd = align(max(smem_cd_l1, smem_cd_l2), 1024)` from `20,480 B` down to **`6,144 B`**, saving **`14,336 B`** of shared memory per SM!

### 2.2 `csrc/apis/mega_mhc.hpp` & `sm100_mega_mhc.hpp`: Fused mHC + Sinkhorn + RMSNorm + Triple `UE8M0` SF Layouts

In `mega_mhc` (`blob 1cbc0c8230f31438b0c17d11285b9cf380540a1b`) and `sm100_mega_mhc.hpp` (`blob 3be09f3fa654f21bfb08e3b51a002dc7c8b07a11`):
1. **Zero-Transpose 3D TMA Descriptors**: `residual [T, hc_mult, H]` and `fn` expose the HC route dimension (`kNumRoutes`) as the 3rd TMA dimension without materializing transposed views;
2. **Shortest-Longest-Task Split-K (`get_num_splits`)**:
   $$\text{num\_splits} = \left\lceil \frac{K_{\text{blocks}}}{\lceil K_{\text{blocks}} / \max(16, \min(32, \lfloor \text{num\_sms} / M_{\text{blocks}} \rfloor)) \rceil} \right\rceil$$
   For `T=4096` (`M_blocks=64`) and `H=7168` (`K_blocks=56`), `get_num_splits` selects **`14` splits** (each handling exactly 4 K-blocks) instead of 16 unequal splits, minimizing barrier traffic while achieving the exact same minimum producer critical path;
3. **Simultaneous Triple `UE8M0` Scaling-Factor Emission**: Packs four `gran_k=32` `UE8M0` exponents per `int32` and writes TMA column-major `y_gemm_sf`, contiguous row-major `y_routed_sf`, and MegaMoE shared-expert `y_shared_sf` in one epilogue pass.

---

## 3. Deterministic Offline Evaluation (`own-offline`)

Verified by [`experiments/deepgemm_mega_moe_mhc_sim/run.py`](../../experiments/deepgemm_mega_moe_mhc_sim/run.py) and [`results.json`](../../experiments/deepgemm_mega_moe_mhc_sim/results.json):

| Workload / `T_rank` (`128` experts) | `E_tokens` / `1-Sigma Covered` | Selected `block_m` (`store_l1 / l2`) | `smem_cd` (`B`) | Stage SMEM (`B`) | TMA `num_stages` | Total SMEM (`<= 232,448 B`) |
|---|---|---|---|---|---|---|
| `T_rank = 16` | `8.0` / `10.83` | **`16`** (`8 / 8`) | `4,096` | `18,448` | **`10` stages** | `218,820 B` |
| `T_rank = 80` | `40.0` / `46.32` | **`64`** (`16 / 16`) | `8,192` | `21,520` | **`9` stages** | `232,372 B` |
| `T_rank = 280` | `140.0` / `151.83` | **`192`** (`32 / 32`) | `16,384` | `29,712` | **`6` stages** | `225,668 B` |
| `T_rank = 400` | `200.0` / `214.14` | **`240`** (**`24 / 8`** asymmetric) | **`6,144`** (`-14,336 B`) | `32,784` | **`5` stages** | `200,820 B` |

For **`MegaMHC` (`T=4096, H=7168, hc_mult=4`)**:
- **HBM Traffic Reduction**: Drops from **`984.38 MiB`** (6 unfused kernels) to **`589.75 MiB`** (**`-40.09%`**);
- **Latency Speedup**: Drops from **`184.79 us`** to **`96.44 us`** on B200 (**`1.92x`** speedup) with **`14`** balanced Split-K partitions.

---

## 4. Architect Decision Guidelines & Falsifiable Questions

1. **When to co-deploy `MegaMHC` + `MegaMoE`**: Whenever serving or training DeepSeek-style MoE models with Hyper-Connections and Shared Experts on SM100 NVLink domains; emitting `y_routed_sf` and `y_shared_sf` inside the `MegaMHC` epilogue eliminates standalone scaling-factor layout kernels.
2. **Falsifiable Follow-Up**: How does `MegaMoE` ring-pool token capacity (`num_ring_tokens`) scale when `num_shared_experts` increases from 1 to 4 under asymmetric network tail latency?
