# DeepGEMM SM100 持久态 `MegaMoE`、非对称 TMA Epilogue 与多头超连接 (`MegaMHC`) 算子架构

> **Language / 语言**: **中文 (Chinese)** | [English (英文版)](deepgemm-sm100-mega-moe-mhc-hyperconnection.en.md)

## 1. 架构背景与核心痛点：MoE 与多头超连接（mHC）在 Blackwell SM100 上的内存墙

在 DeepSeek-V3/V4 规模的稀疏 MoE 与多头超连接（Multi-Head Hyper-Connections, `mHC`）架构中，Blackwell SM100（`232,448 Bytes = 227 KiB` 共享内存/SM）面临两个核心工程瓶颈：
1. **持久态单算子 MoE (`MegaMoE`) 的 SMEM 容量挤压**：在单算子内同时集成跨卡 Dispatch 拉取缓冲（`smem_dispatch_size`）、L1 Gate/Up GEMM、SwiGLU 激活、Amax 归约、L2 Down GEMM 与 Combine 写回（`smem_cd`）时，若在大 Token 块（`block_m = 240`）下采用对称的 C/D 共享内存暂存块（如 `40 × 40`），`smem_cd` 将吞噬 `20,480 B` 共享内存，直接挤压 TMA 多级流水线深度；
2. **多头超连接（`mHC`, `residual [T, hc_mult=4, H=7168]`）的 6 算子 HBM 往返风暴**：未融合的 Pre-Mix 投影、Sinkhorn-Knopp 双随机矩阵迭代、残差流混合、RMSNorm 以及面向 GEMM / Routed-MoE / Shared-MoE 三种不同对齐格式的 FP8 `UE8M0` 缩放因子转换，在 `T=4096, H=7168` 下产生高达 **`984.38 MiB`** 的 HBM 读写。

基于 **`deepseek-ai/DeepGEMM` (`commit 5081f8513af57e61696de4f4f04b0e707ffb503e`)** 最新源码（`csrc/apis/mega_moe.hpp`、`csrc/jit_kernels/heuristics/mega_moe.hpp`、`csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`、`csrc/apis/mega_mhc.hpp` 与 `csrc/jit_kernels/impls/sm100_mega_mhc.hpp`），本文完整推导其启发式分块公式、非对称 Epilogue 与 `MegaMHC` 融合收益。

---

## 2. 一手源码机制拆解 (`implementation`)

### 2.1 `csrc/jit_kernels/heuristics/mega_moe.hpp`：$1\sigma$ 路由裕量分块与非对称 `store_block_m_l1 / l2`

在 `get_block_config_for_mega_moe` (`blob 660223eb73b08365accbdb76d8a6b7d1c61d9414`) 中，DeepGEMM 引入泊松路由标准差（One-Sigma Routing Margin）估算每个本地专家需覆盖的 Token 行数：

$$E_{\text{tokens}} = \frac{T \times R \times K}{E_{\text{total}}}, \qquad N_{\text{covered}} = E_{\text{tokens}} + \sqrt{E_{\text{tokens}}}$$

由于每个额外的 M 分块都会导致权重 Tile 重新加载（约等价于 128 行额外开销），启发式在候选集 `kCandidateBlockM = {16, 32, 64, 128, 192, 240}` 中选择覆盖 $N_{\text{covered}}$ 的最小分块（当 `num_blocks == 1` 且 `N_covered > 192` 但每卡本地专家数 `num_experts / num_ranks < 14` 时，拆分为 2 个较小分块以保证 SM 局部性域负载均衡）。

更关键的是，针对 `MXFP8FP4`（`fp8xfp4` / `fp8xfp8`）Prefill 大分块（`block_m = 240`），DeepGEMM 采用了**非对称 Epilogue 存储分块**：

```cpp
// Use smaller, asymmetric store blocks at FP8xFP4 prefill to save shared memory for sufficient TMA pipeline depth
const int store_block_m_l1 = block_m <= 16 ? 8 : block_m <= 64 ? 16 : block_m <= 192 ? 32 :
                             mma_kind == MmaKind::MXFP8FP4 ? 24 : 40;
const int store_block_m_l2 = (block_m == 240 and mma_kind == MmaKind::MXFP8FP4) ? 8 : store_block_m_l1;
```

同时在 `sm100_fp8_fp4_mega_moe.hpp` (`blob b77ed82e4f438fe95c0bf23fd8035c40969243fe`) 中，L1 SwiGLU 的输出直接原位混叠为 L2 的输入激活张量（`tensor_map_l1_output` 与 `tensor_map_l2_acts` 指向同一 `l2_acts` 环形池），且由于 SwiGLU 后 N 维减半（`block_n / 2 = 64`），TMA Swizzle 模式同步减半（`swizzle_acts_mode / 2 = 64B`）。

### 2.2 `get_pipeline_config_for_mega_moe`：SM100 `232,448 B` 共享内存精确账本

在 `get_pipeline_config_for_mega_moe` 中，每个 CTA 的固定共享内存 `smem_fixed` 与每级 TMA 流水线开销 `smem_size_per_stage` 满足：

$$\text{smem\_cd} = \text{align}_{1024}\left(\max\left(G_{\text{epi}} \cdot M_{\text{store,l1}} \cdot \frac{N}{2} \cdot 2 \cdot b_{\text{mma}},\ G_{\text{epi}} \cdot M_{\text{store,l2}} \cdot N \cdot 2\right)\right)$$

$$\text{smem\_size\_per\_stage} = \frac{M}{2} K b_{\text{mma}} + N K b_{\text{mma}} + M_{\text{sf}} \frac{K}{32} + N_{\text{sf}} \frac{K}{32} + 16$$

$$\text{num\_stages} = \left\lfloor \frac{232448 - \text{smem\_fixed}}{\text{smem\_size\_per\_stage}} \right\rfloor$$

当 `block_m = 240, MXFP8FP4` 时，`store_block_m_l1 = 24`（`smem_cd_l1 = 6,144 B`）且 `store_block_m_l2 = 8`（`smem_cd_l2 = 4,096 B`），使对齐后的 `smem_cd` 从对称 `40/40` 下的 `20,480 B` 降至 **`6,144 B`**（净省 **`14,336 B`** SMEM）。

### 2.3 `csrc/apis/mega_mhc.hpp` & `sm100_mega_mhc.hpp`：多头超连接与三重 `UE8M0` 缩放因子融合

在 `mega_mhc` (`blob 1cbc0c8230f31438b0c17d11285b9cf380540a1b`) 与 `sm100_mega_mhc.hpp` (`blob 3be09f3fa654f21bfb08e3b51a002dc7c8b07a11`) 中：
1. **3D TMA 零转置视图**：`residual [T, hc_mult, H]` 与 `fn [hc_mult*(hc_mult+2), hc_mult*H]` 直接通过 `make_tma_3d_desc` 将超连接路由维 `kNumRoutes = hc_mult` 暴露为第三维，无需显式转置；
2. **最短最长任务 Split-K 启发式 (`get_num_splits`)**：
   $$\text{num\_splits} = \left\lceil \frac{K_{\text{blocks}}}{\lceil K_{\text{blocks}} / \max(16, \min(32, \lfloor \text{num\_sms} / M_{\text{blocks}} \rfloor)) \rceil} \right\rceil$$
   在 `T=4096`（`M_blocks=64`）、`H=7168`（`K_blocks=56`）下，相比固定 `16` 切分（每份最多 $\lceil 56/16 \rceil = 4$ 块），选择 **`14` 切分**（$14 \times 4 = 56$ 块）即可达到完全相同的最短关键路径（4 块/split），同时节省 2 个 Split 的屏障归约开销；
3. **Epilogue 同时输出 3 种 `UE8M0` 缩放因子布局**：`gran_k = 32`，每 4 个 `UE8M0` 打包为一个 `int32`，单次遍历同时写出 TMA 列主序 `y_gemm_sf`、连续行主序 `y_routed_sf` 与 MegaMoE 共享专家布局 `y_shared_sf`。

---

## 3. 原创确定性离线实验 (`own-offline`) 与量化结论

通过 [`experiments/deepgemm_mega_moe_mhc_sim/run.py`](../../experiments/deepgemm_mega_moe_mhc_sim/run.py) 对 `MegaMoE` 与 `MegaMHC` 进行确定性重算（见 [`results.json`](../../experiments/deepgemm_mega_moe_mhc_sim/results.json)）：

| 场景 / 批量维度 | `E_tokens` / `$1\sigma$ Covered` | 选定 `block_m` (`store_l1 / l2`) | `smem_cd` (`B`) | 每级 Stage SMEM (`B`) | TMA `num_stages` | 总 SMEM (`<= 232,448 B`) |
|---|---|---|---|---|---|---|
| `T_rank = 16` (`128` 专家) | `8.0` / `10.83` | **`16`** (`8 / 8`) | `4,096` | `18,448` | **`10` 级** | `218,820 B` |
| `T_rank = 80` (`128` 专家) | `40.0` / `46.32` | **`64`** (`16 / 16`) | `8,192` | `21,520` | **`9` 级** | `232,372 B` |
| `T_rank = 280` (`128` 专家) | `140.0` / `151.83` | **`192`** (`32 / 32`) | `16,384` | `29,712` | **`6` 级** | `225,668 B` |
| `T_rank = 400` (`128` 专家) | `200.0` / `214.14` | **`240`** (**`24 / 8`** 非对称) | **`6,144`** (省 `14,336 B`) | `32,784` | **`5` 级** | `200,820 B` |

在 **`MegaMHC` (`T=4096, H=7168, hc_mult=4`)** 中：
- **HBM 流量削减**：从 6 算子分立管线的 **`984.38 MiB`** 降至单算子 `sm100_mega_mhc` 的 **`589.75 MiB`**（削减 **`40.09%`**）；
- **端到端延迟**：在 B200（`6.8 TB/s` 有效 HBM 带宽）下从 **`184.79 us`** 降至 **`96.44 us`**（**`1.92x`** 提速），动态 Split-K 自动收敛至 **`14`**。

---

## 4. 解决方案架构师决策指南与可证伪问题

1. **何时启用 `MegaMoE` + `MegaMHC` 联合管线？**
   - 在 Blackwell SM100 节点内（NVLink 对称内存域）运行带 Shared Experts 与 Hyper-Connections 的稀疏 MoE 时，通过 `MegaMHC` 直接同时产出 `y_routed_sf` 与 `y_shared_sf`，可零转换直喂 `sm100_fp8_fp4_mega_moe`；
2. **后续可证伪问题**：当 `hc_mult` 从 `4` 扩展到 `8` 时，3D TMA 描述符在 SM100 L2 Cache 中的行命中率与 Sinkhorn 迭代寄存器压力会如何变化？
