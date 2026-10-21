# FlashInfer Cake GLM-5.3 四流 mHC 训练、Blackwell MSA 持久化 K-Major 预填充、FP4 AllReduce 行缩放重叠及 TRTLLM-Gen 多波 KV 切分与 BMM 全序默认选择深度解析

> **演进阶段**：Phase 45 (`priority: 54`) · **上游一数据源**：`flashinfer-ai/flashinfer@130dd2429a` (`#6897`), `2ff6d8a5f2` (`#6873`), `3b65505f40` (`#6904`), `c7d1eacfdc` (`#6875`), `a3f2525e39` (`#6331`), `c1bcf70660` (`#6845`)  
> **确定性复现脚本**：`experiments/flashinfer_cake_mhc_msa_kmajor_multiwave_bmm/run.py --check`

---

## 1. 核心架构发现与量化结论

1. **Cake GLM-5.3 四流 mHC 训练内核 (`#6897`, `130dd2429a`)**：针对 GLM-5.3-Flash 风格四流多头跨流连接（multi-head cross-stream connection, `streams=4, hidden=4096`，展平张量 `u: [T, 16384]`），`flashinfer.cake_mhc_train` 将 RMSNorm、19 步 Sinkhorn-Knopp 双随机矩阵归一化与前/后向融合进 `sm_100a`/`sm_103a`/`sm_107a` 汇编路由中：
   - **前向预备 (`forward_prepare`) 与无状态重算 (`forward_stats`)**：每个 token 仅保存 `256 B`（`64` 个 FP32 标量：`r_inv, u_norm_sq, alpha, a_tilde[4], b_tilde[4], C_tilde[16], a[4], b[4], C[16], dL_da[4], dL_db[4], dL_dC_tilde[16]`），而 `checkpoint` 模式保存 `0 B/token` 并在反向中直接通过 `forward_stats` 重算统计量；
   - **GB300 (`sm_103a`) 与 R200 (`sm_107a`) 实测加速**：在 GB300 上，`T=1024..65536` 相比 `torch.compile` 取得 **`9.31x–15.95x` 前向加速**（`T=65536` 时 `1454.7 us` vs `23197.5 us`）、**`6.61x–8.30x` 反向加速**与 **`9.95x–14.79x` 激活重算加速**（`2823.8 us` vs `41773.9 us`）；在 R200 (`sm_107a`) 上取得 **`7.68x–13.52x` 前向**、**`7.51x–8.60x` 反向**与 **`10.48x–14.28x` 激活重算加速**。
2. **Cake Blackwell MSA 持久化 K-Major 预填充 (`#6873`, `2ff6d8a5f2`) 与 Kimi-K3 Round-2 AttnRes (`#6904`, `3b65505f40`)**：
   - 为 `cake_blackwell_msa` 新增 3 条持久化 K-major 预填充路由（flat GQA-16、paged GQA-16、paged GQA-8）及精确 BF16 归约，每架构覆盖 `37` 个路由阶段，在 GB300（`106` 形状）达到 **`3.39x` 几何平均加速**（paged prefill 峰值 **`5.284x`**，FP8-KV flat decode `B=64, q=4, kv=8192` 达 **`3.229x`**），在 B200（`105` 形状）达 **`3.35x` 几何平均加速**；
   - Round-2 Kimi-K3 AttnRes 将 `133` 个内核程序重打包为 `161` 个多符号库（`sm_100a` `75` + `sm_103a` `86`），在 GB300 全量 `582` 行基准上取得 **`1.0072x` 几何平均加速**（`serving_m5_k5` PDL-off 达 **`1.0888x`**）。
3. **Cake FP4 AllReduce 行缩放在途读取与 TP4 二阶段 SMEM 复用 (`#6875`, `c7d1eacfdc`)**：
   - TP2 融合 AllReduce + RMSNorm 将对端行缩放（row scale）的 `ld.volatile.global.u32` 轮询嵌入第一轮特征扫描中（以 `0x80000000u` 为未就绪哨兵），通过 `__ballot_sync` 跨 warp 投票收敛，消除扫描前自旋等待；
   - TP4 two-shot 路径将第一阶段已校验的值数据包（每行 `3072 B` NVFP4 值 + `384 B` FP8 块缩放）保留在共享内存中，使第二阶段仅需从 Lamport 缓冲区重读 `4 × 4 B = 16 B` 行缩放，将每行 L2 重复读取从 **`3,472 B` 降至 `16 B`（降低 `99.54%`）**。
4. **TRTLLM-Gen Opt-In 4 波 KV 切分解码 (`#6331`, `a3f2525e39`) 与与清单顺序无关的 BMM 默认配置全序比较器 (`#6845`, `c1bcf70660`)**：
   - 当单波次可分配的 KV 切分数 `<= 2` 且 `base_ctas <= 2 * SMs` 时，`FLASHINFER_TRTLLM_GEN_MULTI_WAVE_KV_SPLIT=1` 允许最多 `kMaxNumWavesForKvSplit = 4` 波次，并将 `partialStats` 与 `partialMax` 缓冲按 `std::max(mMultiProcessorCount, totalNumCtas) * mStepQ` 分配（在 B200 `148 SM`、`B=48, H_kv=2` 下从 `2,368` 项扩容至 `9,216` 项，消除多波次 CTA 越界风险）；
   - 修复 `TrtllmGenBatchedGemmRunner::getValidConfigIndices` 中默认配置选择依赖 cubin 清单顺序的缺陷：在调度器层级之前引入 `mNumSlicesForSplitK` 升序比较，将 `Persistent` 调度器的 CTA 数估算改为与候选本身 split-K/cluster 解耦的对称形式，并追加 `mNumStagesA`/`mNumStagesB` 与 `strcmp(mFunctionName)` 决胜规则，在 NVFP4 FC2 (`B=8, M=256, N=4096, K=512`) 上从 `splitK3` (`0.135 ms`) 恢复至不切分内核 (`0.071 ms`, **`1.90x`**)，在 MXFP4×MXFP8 FC1 (`B=128, M=128, N=2048, K=7168`) 上从 `0.718 ms` 恢复至 `0.631 ms` (**`1.14x`**)。

---

## 2. 数学模型与硬件执行机制

### 2.1 四流 mHC 19 步 Sinkhorn-Knopp 双随机混合矩阵 $C$

给定输入 $u \in \mathbb{R}^{T \times 4H}$（$H=4096$），先计算整行均方根倒数 $r^{-1} = (\frac{1}{4H}\sum_{i=1}^{4H} u_i^2 + \varepsilon_{\text{rms}})^{-1/2}$，再通过投影权重 $P \in \mathbb{R}^{24 \times 4H}$ 生成每 token 的 24 维原始控制量 $(\tilde{a}_{\text{raw}} \in \mathbb{R}^4, \tilde{b}_{\text{raw}} \in \mathbb{R}^4, \tilde{C}_{\text{raw}} \in \mathbb{R}^{4 \times 4})$。对于混合矩阵 $\tilde{C}$：
1. 先对每行执行数值稳定 Softmax 并加 $\varepsilon = 10^{-6}$：$C^{(0)}_{ij} = \frac{\exp(\tilde{C}_{ij} - \max_k \tilde{C}_{ik})}{\sum_k \exp(\tilde{C}_{ik} - \max_m \tilde{C}_{im})} + 10^{-6}$；
2. 先执行 1 次列归一化，再连续执行 **19 轮**「行归一化 $\to$ 列归一化」迭代，使最终 $C \in \mathbb{R}^{4 \times 4}$ 严格满足行和与列和均为 $1.0$（误差 $< 10^{-4}$）。

### 2.2 BMM 默认内核严格弱序（Strict Weak Ordering）比较器

设两个有效 BMM 内核配置为 $c_0, c_1$，定义未切分、未聚类的基准 CTA 数为 $N_{\text{CTA}}^{\text{base}}(M, N, B)$：

$$\text{Key}(c) = \bigl(-K_{\text{tile}}, -M_{\text{tile}}, -N_{\text{tile}}, S_{\text{splitK}}, \mathbb{I}[\text{sched}(c) \neq \text{sched}^*(N_{\text{CTA}}^{\text{base}})], -\text{stages}_A, -\text{stages}_B, \text{name}(c)\bigr)$$

该比较器满足反自反性、非对称性与传递性，使默认内核选择完全独立于 cubin 列表顺序。

---

## 3. 实测性能矩阵

| 子系统 / 算子 | 硬件 / 形状 | 基准延迟 / 指标 | 优化后延迟 / 指标 | 加速比 / 收益 |
|---|---|---:|---:|---:|
| Cake mHC Forward (`#6897`) | GB300 (`sm_103a`), `T=65536` | `23,197.5 us` (`torch.compile`) | `1,454.7 us` | **`15.95x`** |
| Cake mHC Checkpoint (`#6897`) | GB300 (`sm_103a`), `T=65536` | `41,773.9 us` (`torch.compile`) | `2,823.8 us` | **`14.79x`** |
| Cake mHC Checkpoint (`#6897`) | R200 (`sm_107a`), `T=65536` | `28,443.5 us` (`torch.compile`) | `1,992.6 us` | **`14.28x`** |
| Cake Blackwell MSA (`#6873`) | GB300 (`106` shapes) / B200 (`105` shapes) | Baseline Cake MSA | Persistent K-Major + FP8-KV | **`3.39x` / `3.35x` geomean** |
| Cake FP4 AllReduce TP4 (`#6875`) | `sm_100a`/`103a`/`107a`, `H=6144` | `3,472 B/row` L2 pass-2 re-read | `16 B/row` L2 pass-2 re-read | **`-99.54%` L2 re-read** |
| TRTLLM-Gen BMM NVFP4 FC2 (`#6845`) | B200, `B=8, M=256, N=4096, K=512` | `0.135 ms` (`splitK3`) | `0.071 ms` (`unsplit`) | **`1.90x`** |
| TRTLLM-Gen BMM MXFP4×MXFP8 (`#6845`) | B200, `B=128, M=128, N=2048, K=7168` | `0.718 ms` | `0.631 ms` | **`1.14x`** |
