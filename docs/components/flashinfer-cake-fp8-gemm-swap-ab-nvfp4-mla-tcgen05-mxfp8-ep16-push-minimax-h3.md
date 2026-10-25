# FlashInfer Cake Grouped FP8 GEMM Swap-AB UE8M0、NVFP4 Sparse MLA Decode tcgen05、MXFP8 MegaMoE EP16 推送邮箱与 MiniMax-H3 SM100/SM103/SM120 架构审计

> English version: [flashinfer-cake-fp8-gemm-swap-ab-nvfp4-mla-tcgen05-mxfp8-ep16-push-minimax-h3.en.md](flashinfer-cake-fp8-gemm-swap-ab-nvfp4-mla-tcgen05-mxfp8-ep16-push-minimax-h3.en.md)
> 证据等级：`implementation` + `author-benchmark`（上游源码/PR 实测）+ `own-offline`（本项目 Python 3.12 标准库确定性仿真器，非本项目 GPU 硬件实测）

## 1. 核心架构问题与源码演进路径

Phase 84（`priority: 93`）针对 `flashinfer-ai/flashinfer` 在 `sm_100a`（B200/GB200）、`sm_103a`（GB300）与 `sm_120a`（RTX 5090 / RTX PRO 6000 Blackwell Server Edition）上的 6 个第一手内核提交进行了深入审计：

1. **`cake_grouped_fp8_gemm` 打包 UE8M0 块缩放 Swap-AB 路由（[#7342](https://github.com/flashinfer-ai/flashinfer/commit/84f3bb30b29e2a927bfdacdef2bbe7eca5830fc0)）**：
   - 在填充连续布局（padded contiguous layout）Grouped FP8 GEMM 中，为 2 的幂次打包 `UE8M0` 块缩放新增 `swap`、`swap_run32`（针对 32-row group 对齐）与 `swap_carry` 三套 Swap-AB 程序（同时支持 `sm_100a` 与 `sm_103a`）。
   - 在稀疏小批次路由（如 `M=16000, N=512, K=4096, groups=512, align=32, 2x10 uniform`）下，`swap_run32` 将 B200 延迟从 `42.30 us` 降至 `24.58 us`（`new/shipped = 0.581`，**1.721x** 加速），GB300 slow 从 `40.56 us` 降至 `23.49 us`（`0.579`，**1.727x** 加速）。
2. **`nvfp4_sparse_mla_decode(..., backend="cake")` 35 套 `tcgen05` 程序（[#7335](https://github.com/flashinfer-ai/flashinfer/commit/a8ec053888bdb597e9a0a3ca9ae57dfe1fc8b900)）**：
   - 算子接收 E4M3 查询 `[T, 16, 576]` 与 `int32` top-k 行索引（`-1` 表示空槽），寻址 352 字节 `nvfp4_ds_mla` KV Cache 行并输出 BF16 `[T, 16, 512]`。
   - 原本在 GB200 上 4-CTA cluster 的 `mma.sync` 路径受限于 L1/Shared Memory 流量（占比 69%，其中 63% wavefront 为写一次、被 QK 与 PV 读两次的 FP16 Key Tile）。新路径用 **35 套 `tcgen05` 程序**（覆盖 `C in {1, 2, 3, 4, 5, 6, 8}` 共 7 种每 Token CTA 簇大小 × `{64, 128, 192, 256, any}` 5 种每 CTA Key 切分上限）替代 7 套 `mma.sync` 程序，将 Key Tile 驻留在 Blackwell **Tensor Memory (TMEM)** 中。
   - Host Planner 仅筛选可将所有 Token 簇装入单个 Wave 的 `C`，并选择每 CTA 负载最小的配置：B200 39 组场景几何平均加速 **1.131x**（最高 **1.305x**，`W=128, topk=2048` 从 `63.94 us` 降至 `48.99 us`）；GB300 40 组场景几何平均加速 **1.152x**（最高 **1.375x**，`61.60 us -> 44.80 us`）。
3. **`CakeMxfp8MegaMoeEp16` 推送式对称邮箱与异步代理内存栅栏（[#7350](https://github.com/flashinfer-ai/flashinfer/commit/38b6198055ca44baa10600e62b9c513b59ead0c8)）**：
   - 面向 16-Rank Expert Parallelism（`hidden=3072, inter=5120, 512 experts, 32/rank, top-8`，BF16 激活 + MXFP8 E4M3/UE8M0 权重）：
   - 将 Dispatch 从跨 NVLink 拉取（Pull）改为**跨 Rank 屏障前推送（Push）**：对称内存缓冲区按源 Rank 划分为 `[16, T, 3072]` 行槽位与 `[16, T, 8]` 路由槽位（`T=64` 时每 Rank 约 `6 MiB`），屏障后本地直接读取；
   - 每个 CTA 直接等待本 Rank 的屏障标志（消除 CTA 0 的二次 GPU-scope 转发及逐线程 `fence.sc.sys`）；
   - TMA Warp 在 Dispatch 完成与每次 FC1 完成等待后显式执行 `fence.proxy.async.global`（CuTe DSL 后端同步补充 Shared Memory Proxy Fence），确保通用存储写入对 TMA 异步引擎可见。16x GB300 (NVL72) 端到端延迟降至 `295.4–299.3 us`（**1.014x–1.019x** 加速）。
4. **MiniMax-H3 SM100/SM103/SM120 投影与变长注意力优化（[#7314](https://github.com/flashinfer-ai/flashinfer/commit/b4125442a53382fad96d08bbf3eda580c9d58a02)、[#7333](https://github.com/flashinfer-ai/flashinfer/commit/53ecbde224f45741a2aa4a0b928a330a502ab15e)、[#7334](https://github.com/flashinfer-ai/flashinfer/commit/b8ed186df31b1ad175d6b0b4a47120542c48279d)）**：
   - SM100/SM103 `minimax_h3_out_proj_nvfp4` 对最后不满半个 Wave 的尾部 Tile 启用 `256x128` 半宽 N-Split 单元，在 GB200 上取得最高 **1.052x**（geomean **1.042x**）、GB300 上最高 **1.041x**（geomean **1.021x**）加速且逐比特等价；
   - SM120 `minimax_h3_fp8_out_proj` 改为每 Swath 步进单个 M Tile，在 RTX PRO 6000 Blackwell 上实现 **1.006x–1.011x** 加速；
   - SM120 `minimax_h3_sm120_varlen_attention_fp8` 引入提前 PV 移交（Early PV Hand-off）与下一计算单元预取（Next-Unit Prefetch），在 RTX 5090 上加速 **1.014x–1.055x**，在 RTX PRO 6000 上加速 **1.030x–1.041x**。

## 2. 关键公式与带宽/容量计算

- **MXFP8 MegaMoE EP16 每 Rank 对称邮箱内存占用**：
  $$\text{Bytes}_{\text{mailbox}}(W, T, H, K) = W \cdot T \cdot H \cdot 2 + W \cdot T \cdot K \cdot 8$$
  当 $W=16, T=64, H=3072, K=8$ 时，行邮箱为 $16 \times 64 \times 3072 \times 2 = 6,291,456\text{ B} = 6.0\text{ MiB}$，路由邮箱为 $16 \times 64 \times 8 \times 8 = 65,536\text{ B} = 64\text{ KiB}$。
- **NVFP4 Sparse MLA 单 Wave 簇选择约束**：
  $$\text{ActiveClusters}(C) = \left\lfloor \frac{\text{SM}_{\text{count}}}{C} \right\rfloor \ge T, \quad C^* = \arg\min_{C \in \{1,2,3,4,5,6,8\}, \text{ActiveClusters}(C) \ge T} \left\lceil \frac{\text{topk}}{C} \right\rceil$$

## 3. 可复现实验与架构师练习

- 运行命令：`python experiments/flashinfer_cake_fp8_gemm_swap_ab_nvfp4_mla_tcgen05_mxfp8_ep16_push_minimax_h3/run.py`
- 关联证据 ID：`ev-flashinfer-cake-fp8-gemm-swap-ab-nvfp4-mla-tcgen05-mxfp8-ep16-push-minimax-h3-2026`
- **架构师设计练习**：计算当 `T=35` 时在 GB200（148 SMs）与 GB300（152 SMs）上单 Wave 允许的最大簇大小 $C$，并解释为何 `T=36` 在 GB300 上可维持 $C=4$（$36 \times 4 = 144 \le 152$）而在 B200（148 SMs）上 `T=37` 是 $C=4$ 的上界。
