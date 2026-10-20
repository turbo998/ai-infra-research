# DeepGEMM SM100 `MegaMoE` 与 `MegaMHC` 超连接算子算例 / DeepGEMM SM100 MegaMoE & MegaMHC Simulation (`deepgemm_mega_moe_mhc_sim`)

## 中文概述 (Chinese Overview)

本实验基于 `deepseek-ai/DeepGEMM` (`commit 5081f8513af57e61696de4f4f04b0e707ffb503e`) 源码中的 `csrc/jit_kernels/heuristics/mega_moe.hpp`、`csrc/apis/mega_moe.hpp`、`csrc/apis/mega_mhc.hpp` 与 `csrc/jit_kernels/impls/sm100_mega_mhc.hpp`，复现并验证：
1. **`get_block_config_for_mega_moe` $1\sigma$ 路由裕量分块启发式**：计算 `num_covered_tokens = E + sqrt(E)`，在 `num_tokens = [16, 80, 280, 400]` 下自适应选择 `block_m = [16, 64, 192, 240]`，对应 SM100 (`232,448 B` SMEM) TMA 流水线深度 `num_stages = [10, 9, 6, 5]`。
2. **`MXFP8FP4` 非对称 Epilogue Store 分块 (`store_block_m_l1=24, store_block_m_l2=8`)**：在 `block_m=240` 时将 `smem_cd` 从对称 `40/40` 的 `20,480 B` 压缩至 `6,144 B`（节省 **`14,336 B`** 共享内存），保障持久态 `MegaMoE` 在 `232,448 B` 容量内的 `5` 级 TMA 流水线。
3. **`sm100_mega_mhc` 多头超连接（mHC）融合算子**：单算子融合 `residual [T, 4, 7168]` 超连接混合、Sinkhorn-Knopp 双随机迭代、RMSNorm 与三种 FP8 `UE8M0` (`gran_k=32`) 缩放因子布局（`y_gemm_sf`、`y_routed_sf`、`y_shared_sf`），动态 Split-K `14` 切分，HBM 流量降低 **`40.09%`**（`984.38 -> 589.75 MiB`），延迟从 `184.79 us` 降至 `96.44 us`（**`1.92x`** 提速）。

## English Overview

This deterministic Python 3.12 standard-library experiment models `deepseek-ai/DeepGEMM` (`commit 5081f8513af57e61696de4f4f04b0e707ffb503e`):
1. **One-Sigma Routing Margin & Tile Heuristics (`get_block_config_for_mega_moe`)**: Computes `num_covered_tokens = E + sqrt(E)` and selects `block_m = [16, 64, 192, 240]` with `num_stages = [10, 9, 6, 5]` across SM100's `232,448 B` shared memory budget.
2. **Asymmetric Epilogue Store Blocks (`store_block_m_l1=24, store_block_m_l2=8`)**: At `block_m=240` under `MXFP8FP4`, asymmetric staging shrinks `smem_cd` from `20,480 B` to `6,144 B` (saving **`14,336 B`** of SMEM per SM).
3. **Fused Multi-Head Hyper-Connections (`sm100_mega_mhc`)**: Fuses normal/shifted mHC (`hc_mult=4`), Sinkhorn-Knopp normalization, RMSNorm, and simultaneous emission of three `UE8M0` (`gran_k=32`) scaling-factor layouts (`y_gemm_sf`, `y_routed_sf`, `y_shared_sf`) with dynamic Split-K (`14`), reducing HBM traffic by **`40.09%`** (`984.38 -> 589.75 MiB`) and achieving a **`1.92x`** speedup (`184.79 -> 96.44 us`).

## 运行与校验 / Run & Verify

```powershell
python experiments\deepgemm_mega_moe_mhc_sim\run.py --check
```
