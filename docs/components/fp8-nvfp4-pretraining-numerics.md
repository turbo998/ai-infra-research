# FP8 细粒度分块缩放与 Blackwell NVFP4 预训练数值稳定性：源码契约、RHT 离群平滑与随机舍入误差模型

## 1. 核心架构问题与结论摘要

在大规模基础模型（LLM / MoE）预训练与强化学习后训练（RLVR）中，将算术精度从 `BF16` 压降至 `FP8`（`E4M3` / `E5M2`）与 Blackwell `NVFP4`（`E2M1`）可将 Tensor Core 峰值算力提升 **2x–4x**，并同比例削减激活显存、权重 AllGather 通信量与 HBM 带宽压力。然而，Transformer 残差流中天然存在的**离群通道（Massive Activation / Gradient Outliers，幅值达正常通道的 `10x–200x`）**与极窄尾数网格（`E2M1` 仅有 `1` 位尾数、正半轴仅 `8` 个离散点）构成了严峻的数值稳定性冲突：

1. **Per-Tensor `DelayedScaling` 的非离群通道下溢（Underflow）与 Amax 滞后跳变**：在 `NVIDIA/TransformerEngine` 的 `DelayedScaling` 中，整个张量共享单一 `FP32` 缩放因子 $s = \text{FP8\_MAX} / \text{amax}_{\text{history}}$（`E4M3` 下 $\text{FP8\_MAX} = 448.0$）。当单个 token 或离群通道出现 `100x` 尖峰时，全局缩放因子被拉低 `100x`，导致其余 `99.9%` 的正常激活/梯度通道落入 `E4M3` 非规格化数（Subnormal）或直接下溢归零，在本仓离线数值实验（`experiments/fp8_nvfp4_numerics/`）中非离群通道信噪比（`SQNR`）从 **`31.94 dB` 暴跌至 `21.49 dB`**（`200x` 离群下跌至 **`15.47 dB`**）。
2. **DeepSeek `1x128` / `128x128` 细粒度分块缩放与 WGMMA FP32 晋升累加（`DeepGEMM`）**：`deepseek-ai/DeepGEMM`（`sm90_fp8_gemm_1d2d.cuh`）在编译期通过 `DG_STATIC_ASSERT(BLOCK_K == 128)` 强制对激活/梯度按 `1x128` 子块、对权重按 `128x128` 子块独立计算缩放因子，并在 Hopper `WGMMA` 每完成 `128` 次乘加后通过 CUDA Core 晋升至完整 `FP32` 寄存器累加（解决 Hopper FP8 Tensor Core 内部累加器仅 `13-bit` 有效尾数位宽导致的截断漂移）。实验证明：在 `100x` 离群通道下，`1x128` 分块缩放保持 **`31.94 dB`** 非离群 `SQNR`（比 `DelayedScaling` 高出 **`+10.45 dB`**），且每 `128` 步晋升 `FP32` 将 `K=4096` 内积相对累加误差从 **`0.0346%` 降至 `0.0056%`**（降低 **`6.18x`**）。
3. **Blackwell `NVFP4BlockScaling` 双层缩放 + `16x16` 随机哈达玛变换（RHT）+ 随机舍入（SR）**：`NVIDIA/TransformerEngine`（`nvfp4.rst` / `core_nvfp4.cuh` / `recipe/__init__.py`）在 `SM100` 上采用双层缩放 $x = x_{\text{e2m1}} \cdot s_{\text{block, e4m3}} \cdot s_{\text{global, fp32}}$（每 `16` 个元素共享 1 个 `FP8 E4M3` 块缩放因子，全张量共享 1 个 `FP32` 全局缩放因子）。为攻克 `4-bit E2M1` 的量化崩塌，`NVFP4BlockScaling` 引入三项协同机制：
   - **权重 `16x16` 2D 分块量化（`fp4_2d_quantization=True`）**：确保前向行主序（Rowwise）量化权重与反向转置列主序（Columnwise）量化权重在数值上严格等价（避免 MXFP8/1D 分块中正反向看到不同量化权重的振荡）。
   - **`16x16` 随机哈达玛变换（`RHT`, $H = \frac{1}{\sqrt{16}} S_{16} H_{16}$）**：在敏感的 `WGRAD` GEMM（仅支持 `TN` 布局，需要对输入激活与上游梯度做列主序量化）前执行正交旋转 $x' = x H$，利用 $(A H)(H^T B) = A B$ 在内积中自动抵消旋转，将块内峰值因子（Crest Factor）从 **`3.048` 压平至 `1.410`**，`100x` 离群下 `WGRAD` 内积 `SQNR` 从 **`18.19 dB` 跃升至 `35.06 dB`（净增益 `+16.87 dB`）**。
   - **硬件随机舍入（`Stochastic Rounding, SR`）**：在反向梯度量化时按距离反比概率舍入至相邻两个 `E2M1` 网格点，确保 $\mathbb{E}[Q(g)] = g$。实验表明，对落入网格间隙的微弱梯度（`g = 0.18`，块缩放 `0.5`），确定性最近偶数舍入（`RTN`）产生 **`38.89%`** 的恒定系统性偏差，而 `SR` 在 `4,096` 步累加后的均值偏差仅为 **`1.25%`**（降低 **`31.1x`**）。

---

## 2. 一手源码实现路径与数据布局核验

### 2.1 `NVIDIA/TransformerEngine` 配方契约与 `NVFP4` 双层缩放

在 `NVIDIA/TransformerEngine`（commit `eb3c860a4d3394b7eb5e21dba3021fa10782a52e`）中：

- **`transformer_engine/common/recipe/__init__.py`（blob `f9abf9de87a91f97bb2e8d11ef1533191b905d25`）**：
  - `Format` 枚举定义了各低精度格式的最大可表示正数：`E2M1 = 6.0`、`E4M3 = 448.0`、`E5M2 = 57344.0`、`UE5M3 = 114688.0`。
  - `DelayedScaling` 默认维护 `amax_history_len = 1024` 步历史窗口，通过 `new_scaling_factor = (FP8_MAX / amax) / (2 ^ margin)` 计算下一步缩放因子，并在分布式组内执行 `reduce_amax` 同步。
  - `Float8BlockScaling` 强制要求 `fp8_gemm_fprop`、`fp8_gemm_dgrad`、`fp8_gemm_wgrad` 全部启用 `MMParams(use_split_accumulator=True)`，并在 `__post_init__` 中断言禁止 `2D x 2D` 分块 GEMM（デフォルト为激活 `x_block_scaling_dim=1`、权重 `w_block_scaling_dim=2`、梯度 `grad_block_scaling_dim=1`）。
  - `NVFP4BlockScaling` 在 `__post_init__` 中分别配置三路量化器参数（`QParams`）：
    - `fp4_quant_fwd_inp`: `random_hadamard_transform=True`, `stochastic_rounding=False`, `fp4_2d_quantization=False`
    - `fp4_quant_fwd_weight`: `random_hadamard_transform=False`, `stochastic_rounding=False`, `fp4_2d_quantization=True`
    - `fp4_quant_bwd_grad`: `random_hadamard_transform=True`, `stochastic_rounding=True`, `fp4_2d_quantization=False`
  - 此外，针对 RL 与后训练场景，`NVFP4BlockScaling` 提供了自适应块上限 `nvfp4_4over6`（在每个 16 元素块上对比映射到 `4.0` 与映射到 `6.0` 的 `MAE`/`MSE` 误差并择优存储，以加密 `[-4, 4]` 区间内的量化分辨率）。

- **`transformer_engine/common/cast/nvfp4/core_nvfp4.cuh`（blob `167703b7fb968b85ad1214dd700699d52beae456`）与 `docs/features/low_precision_training/nvfp4/nvfp4.rst`（blob `26d798651d58fa5a4a3b0e9037b45f532252273c`）**：
  - 全局与局部缩放因子耦合公式：
    $$s_{\text{global}} = \frac{\text{global\_amax}}{\text{FP8\_MAX} \times \text{FP4\_MAX}} = \frac{\text{global\_amax}}{448.0 \times 6.0}, \qquad s_{\text{block}} = \text{ FP8\_E4M3}\!\left(\frac{\text{block\_amax} / 6.0}{s_{\text{global}}}\right)$$
  - `core_nvfp4.cuh` 中的 `compute_decoding_scaling_factor` 将 `block_decode_scale = block_amax * (1.0f / 6.0f)` 与 `global_encode_scale`（即 $1 / s_{\text{global}}$）相乘后直接窄化转换为 `ScaleType`（`fp8e4m3` 或 CUDA 13.4+ 支持的 `fp8ue5m3`）。
  - **转置与硬件对齐填充**：由于 Blackwell `NVFP4` Tensor Core GEMM 仅支持 `TN` 布局，Columnwise（列主序）量化数据与缩放因子直接以转置形状 `[B, A]` 与 `[B, A/16]` 存储，且缩放因子张量的第 1 维必须向上对齐至 `128` 的倍数、第 2 维对齐至 `4` 的倍数（`[roundup(A, 128), roundup(B/16, 4)]`）以满足 TMA（Tensor Memory Accelerator）描述符对齐契约。

### 2.2 `deepseek-ai/DeepGEMM` SM90 / SM100 分块缩放内核对比

在 `deepseek-ai/DeepGEMM`（commit `1fc9c66975ee322d628b3288534ca0e9ce12a64d`）中：

- **Hopper SM90 1D-2D 分块缩放（`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`，blob `5851b6dad1d18b25dedf12afc23aa89ec668e373`）**：
  - 通过 `DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling")` 锁定 K 维 `128` 粒度。
  - 共享内存同时通过 TMA 加载 `SMEM_A`（`BLOCK_M * 128 * sizeof(__nv_fp8_e4m3)`）、`SMEM_B`（`BLOCK_N * 128 * sizeof(__nv_fp8_e4m3)`）以及激活侧每行 1 个 `FP32` 缩放因子 `SMEM_SFA`（按 `128B` 对齐）。每个 `BLOCK_K = 128` 迭代步完成 `WGMMA` 后，在 Epilogue / Math Warp 中将 `WGMMA` 累加器乘以 $s_{A} \cdot s_{B}$ 并累加至 CUDA Core `FP32` 寄存器。
- **Blackwell SM100 原生微块缩放（`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`，blob `082d4d3e8fc368a116d462b83de8839ebc1940ac`）**：
  - 采用 Blackwell 第五代 Tensor Core `tcgen05`（UMMA）指令与 전용 Tensor Memory（`TMEM::Allocator1Sm` / `Allocator2Sm`）。
  - 当左、右操作数均为 `cutlass::float_e2m1_t`（`FP4xFP4`）时，启用专用 `MXF4 MMA`（紧凑打包 4-bit 操作数，`UMMA_K = 64`）；当为 `FP8xFP4` 混合精度（如权重 FP4、激活 FP8）时，启用 `MXF8F6F4 MMA`（`UMMA_K = 32`），在 Tensor Core 硬件流水线内部直接完成块缩放因子乘法，无需软件循环晋升到 CUDA Core。

```
+---------------------------------------------------------------------------------------------------+
| Hopper SM90 DeepGEMM (1x128 x 128x128) vs Blackwell SM100 NVFP4 (tcgen05 1x16 Hierarchical)       |
+---------------------------------------------------------------------------------------------------+
| [SM90 FP8 1D2D]                                                                                   |
|   HBM -> TMA -> SMEM (A_fp8[M,128], B_fp8[N,128], SFA_fp32[M])                                    |
|       -> WGMMA (128 K-steps, 13-bit fast accum) -> CUDA Core FFMA Promote: Acc_fp32 += WGMMA*sA*sB|
|                                                                                                   |
| [SM100 NVFP4 + RHT + SR]                                                                          |
|   BF16 Grad/Act -> 16x16 RHT (x' = x * S_16 * H_16 / 4) -> SR Cast to E2M1 + E4M3 s_block (1x16)  |
|   HBM -> TMA -> SMEM -> tcgen05 UMMA (UMMA_K=64, HW block-scale multiply in Tensor Memory TMEM)   |
|       -> Epilogue dequantize by (s_global_A * s_global_B) -> BF16/FP32 Output                     |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. 原创 CPU/Offline 数值稳定性实验与反证分析

本仓在 `experiments/fp8_nvfp4_numerics/run.py` 中实现了零第三方依赖的确定性数值仿真器，精确复刻了 `FP8 E4M3`（含 `2^-6` 非规格化区间至 `448.0`）、`NVFP4 E2M1`（`{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`）、`16x16` 归一化 Sylvester-Hadamard 正交变换、硬件随机舍入（`SR`）以及 `WGMMA` 有限尾数累加晋升。

### 3.1 跨离群比（`10x–200x`）非离群通道 SQNR 与 WGRAD RHT 收益矩阵

| 离群通道幅值比 (`outlier_ratio`) | Per-Tensor FP8 `DelayedScaling` 非离群 SQNR (`dB`) | DeepSeek FP8 `1x128` 非离群 SQNR (`dB`) | OCP `MXFP8 1x32` (`UE8M0`) 非离群 SQNR (`dB`) | NVFP4 `1x16` (Naive) 非离群 SQNR (`dB`) | NVFP4 `1x16 + 4over6` 非离群 SQNR (`dB`) | `WGRAD` NVFP4 Naive 内积 SQNR (`dB`) | `WGRAD` NVFP4 + `16x16 RHT` 内积 SQNR (`dB`) | 16元素块峰值因子 (`Raw -> RHT`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`10x`** | `31.27 dB` | **`31.94 dB`** | `31.58 dB` | `20.34 dB` | `21.01 dB` | `18.18 dB` | **`25.83 dB`** (`+7.65 dB`) | `3.048 -> 1.411` |
| **`25x`** | `30.66 dB` | **`31.94 dB`** | `31.58 dB` | `20.43 dB` | `20.94 dB` | `18.19 dB` | **`32.90 dB`** (`+14.71 dB`) | `3.048 -> 1.410` |
| **`50x`** | `27.38 dB` | **`31.94 dB`** | `31.58 dB` | `20.39 dB` | `20.92 dB` | `18.19 dB` | **`40.79 dB`** (`+22.60 dB`) | `3.048 -> 1.410` |
| **`100x`** | `21.49 dB` | **`31.94 dB`** | `31.58 dB` | `20.18 dB` | `20.35 dB` | `18.19 dB` | **`35.06 dB`** (**`+16.87 dB`**) | `3.048 -> 1.410` |
| **`200x`** | `15.47 dB` | **`31.94 dB`** | `31.58 dB` | `19.46 dB` | `19.19 dB` | `18.19 dB` | **`35.39 dB`** (`+17.20 dB`) | `3.048 -> 1.410` |

### 3.2 关键机制反证与边界剖析

1. **为什么 `NVFP4` 在 `200x` 离群下仍会出现非离群块微弱退化（`20.43 dB -> 19.46 dB`）？**
   - 尽管 `NVFP4` 每 `16` 个元素拥有独立块缩放因子 $s_{\text{block}}$，但 $s_{\text{block}}$ 本身存储在 **`FP8 E4M3`** 中（动态范围上限 `448.0`），并与全张量共享的 $s_{\text{global}} = \text{global\_amax} / (448 \times 6)$ 耦合。当全张量某处出现极端尖峰使 $\text{global\_amax} / \text{block\_amax} > 200$ 时，正常块的 $s_{\text{block}} = (\text{block\_amax}/6)/s_{\text{global}}$ 会逼近 `E4M3` 的小数值区，导致块缩放因子自身的量化步长变粗。这也是分布式序列并行（Sequence Parallelism）AllGather 前必须同步 `global_amax`、且需要在注意力投影层配合 QK-Norm / Clamp 控制极端激活尖峰的根本原因。
2. **为什么 `Stochastic Rounding (SR)` 只用于反向梯度（`bwd_grad`）而不默认用于前向权重与激活？**
   - 从上表 `results.json` 可见，单次采样的随机舍入（`nvfp4_rht_sr_non_outlier_sqnr_db = 17.14 dB`）相比确定性最近舍入（`RTN = 20.58 dB`）在**单步瞬时方差**上高出约 `3.4 dB`（因为概率跳变引入了零均值白噪声）。前向推理/激活需要最小化单步 MSE，故采用 `RTN`；而反向梯度在训练数千步中持续累加，确定性 `RTN` 的截断死区会产生 **`38.89%`** 的方向性系统偏差（导致小梯度参数永久停滞），`SR` 则利用无偏性（$\mathbb{E}[Q(g)] = g$）在 `4,096` 步内将累积偏差压至 **`1.25%`**。

---

## 4. 解决方案架构师选型矩阵与排障练习

### 4.1 硬件代际与低精度预训练配方选型表

| 架构维度 | Hopper H100 / H200 (`SM90`) | Blackwell B200 / GB200 (`SM100`) | 混合精度回退层（Attention / MoE Router / Embedding） |
| --- | --- | --- | --- |
| **首选算术配方** | DeepSeek `1x128` / `128x128` `Float8BlockScaling`（优于 `DelayedScaling`） | `NVFP4BlockScaling`（`1x16 E2M1 + E4M3 + FP32`）或 `MXFP8BlockScaling` | 保持 `BF16` / `FP32`（`NVFP4` 默认 `fp8_dpa=False, fp8_mha=False`） |
| **显存与缩放元数据开销** | `1 Byte/elem` + `1/128 FP32` scale（`8.25 bits/elem`） | `0.5 Byte/elem` + `1/16 FP8` scale（`4.5 bits/elem`，节省 `45.5%`） | `2 Bytes/elem`（`BF16`）或 `4 Bytes/elem`（Master Weight / Router logits） |
| **核心稳定性保护机制** | `use_split_accumulator=True` + 每 `128` K-step 晋升 CUDA Core `FP32` | 权重 `16x16` 2D 量化 + 激活/梯度 `16x16 RHT` + 梯度硬件 `SR` | `CustomRecipe` 对首尾层（`lm_head`）与高敏感层返回 `IdentityQuantizer` |

### 4.2 架构师动手计算练习

- **练习问题**：某 `70B` Dense 模型在 Blackwell `GB200` 上使用 `NVFP4BlockScaling` 训练，某隐藏层维度为 `M = 4096`（token 数）、`K = 8192`（隐藏维）。请计算该激活张量在 Rowwise 与 Columnwise 双路量化下的 `E2M1` 数据字节数与对齐填充后的 `E4M3` 块缩放因子字节数总和，并与 `BF16` 原张量对比。
- **参考解答**：
  1. `BF16` 原始张量大小：$4096 \times 8192 \times 2\text{ B} = 67,108,864\text{ B} = 64.0\text{ MiB}$。
  2. Rowwise `NVFP4` 数据大小：$4096 \times 8192 \times 0.5\text{ B} = 16.0\text{ MiB}$；Rowwise `E4M3` 缩放因子形状为 `[roundup(4096, 128), roundup(8192/16, 4)] = [4096, 512]`，占用 $4096 \times 512 \times 1\text{ B} = 2.0\text{ MiB}$。
  3. Columnwise（供 `WGRAD` 使用）同样占用 `16.0 MiB` 数据 + `2.0 MiB` 转置缩放因子。双路合计仅 `36.0 MiB`（单路通信仅 `18.0 MiB`，相当于 `BF16` 的 **`28.1%`**）。

---

## 5. 证据链与后续可证伪问题

- **证据链索引**：
  - `ai-infra-fp8-nvfp4-001`：`NVIDIA/TransformerEngine` `nvfp4.rst` / `core_nvfp4.cuh` / `recipe/__init__.py`（commit `eb3c860a4d3394b7eb5e21dba3021fa10782a52e`）——核验 `NVFP4BlockScaling` 双层缩放公式、2D 权重 `16x16` 量化、`16x16 RHT` 与 `Stochastic Rounding` 实现契约。
  - `ai-infra-fp8-nvfp4-002`：`deepseek-ai/DeepGEMM` `sm90_fp8_gemm_1d2d.cuh` / `sm100_fp8_fp4_gemm_1d1d.cuh`（commit `1fc9c66975ee322d628b3288534ca0e9ce12a64d`）——核验 `BLOCK_K == 128` 细粒度分块缩放与 `SM100 tcgen05` `MXF4`/`MXF8F6F4` 硬件指令路径。
  - `ai-infra-fp8-nvfp4-003`：本仓可复现离线算例 `experiments/fp8_nvfp4_numerics/run.py` 与 `results.json`（`own-offline`）。
- **后续可证伪问题**：在超大规模 MoE 路由门控后，当单个 Expert 分到的有效 token 数 $M_e < 16$ 时，`NVFP4` Columnwise 量化在 `M` 维度的 `16x16 RHT` 零填充（Zero-Padding）是否会稀释非零 token 的有效动态范围，此时切换到 `nvfp4_4over6`（关闭 RHT）能否在短序列/稀疏专家块上获得更低的验证集 Perplexity？
