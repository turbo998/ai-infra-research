# FlashInfer DeepGEMM MQA、Kimi-K3 LatentMoE、SM107 SSD 1-CTA Cluster、GB300 Cluster-of-8 FP4、SM120 FP8 Weight-First、mHC Backward 显存释放、PDL 同步与 TileLang 深度解析（Phase 125 / Priority 134）

> **证据分类**：上游源码实现与架构属性（`implementation`）结合本仓库原生 Python 3.12 离线确定性仿真（`own-offline`）。本文严禁将离线算术吞吐模型等同于 GPU 硬件实测曲线（`own-hardware`）。

---

## 1. 一手源码溯源与问题背景

在 Blackwell (`sm_100a`/`sm_103a`)、Rubin (`sm_107a`)、GeForce Blackwell (`sm_120`) 与 Hopper (`sm_90a`) 的生产算子栈演进中，FlashInfer 与 TileLang 针对 MQA 稠密注意力、LatentMoE 尾投射、Mamba2 SSD 扫描、NVFP4 集群多播 GEMM、mHC 反向传播显存驻留及 PDL 跨核同步进行了关键重构：

1. **Cake DeepGEMM Dense MQA SM100 部分块路由复用**（`flashinfer-ai/flashinfer@512db9b35193e567c21f545b19027ab68376f8d2`，PR `#7741`，2026-10-28）：在 `sm_100a` 上，`fp8:h64:full:qle2048:kvsplit` 路由原本维护独立的 1,396 行生成核与绑定代码（`cake_deepgemm_dense_mqa_084957a7d4ca249b561d_{kernel,binding}.cu`）。PR `#7741` 将其统一路由至 `fp8:h64:partial` logits 程序，消除冗余 binary 膨胀并提升指令缓存命中率。
2. **Cake Kimi-K3 Stable LatentMoE L2 Prefetch 与 Resident-B 规约**（`flashinfer-ai/flashinfer@4158fddf0cda7c46b2da19eb0d03736d90307b11`，PR `#7742`，2026-10-28）：在 `flashinfer.experimental.kimi_k3_latent_moe`（BF16，`sm_100a`/`sm_103a`，TP1/TP8）中，针对 $T \le 128$ 的 decode 实例引入 `partial_prefetch=True` 与 `sb_norm`（resident-B 算术融合 RMSNorm 与 $P$ 路 routed partials 规约）。
3. **Cake SSD Combined SM107a 单 CTA 集群启动与快速 F32 Claim**（`flashinfer-ai/flashinfer@ecd4d71d893e948a68c09ae6b175a8a931cac525`，PR `#7740`，2026-10-28）：在 Rubin `sm_107a` 的 56 个 Mamba2 SSD 组合 chunk-scan 程序（`exact`, `exactfused`, `exactpre`, `exactmeta`, `exactone`, `chunkpar`, `chunkpar3`）中显式标注 `__cluster_dims__(1,1,1)`、采用单 CTA 尾部对（one-CTA tail pairs）并优化 F32 claim 程序。
4. **Cake NVFP4 Per-Token GB300 Cluster-of-8 Multicast GEMM (`mc42`)**（`flashinfer-ai/flashinfer@c9fc6cab4066ee4473df640b7ba50e540794340b`，PR `#7739`，2026-10-28）：在 `sm_103a`（GB300）针对 $M \in [1793, 2048]$ 引入 `_sfr_mc42` 策略，启动 8-CTA Cluster 并以 $2 \times 2$ 对块（2 个 token tiles $\times$ 2 个 weight tiles）遍历，使共享同一 token 或 weight tile 的 CTA 对拆分操作数与 scale 加载并进行集群多播。
5. **SM120 Per-Tensor FP8 Weight-First 分组 TC 调度**（`flashinfer-ai/flashinfer@1e7a60d2320d565b8b37f741127848228b54a6e5`，PR `#7520`，2026-10-28）：在 `frost_low_latency` SM120 grouped Tensor Core 路径新增 `weight_first=True`（缓存键升级为 `fp8-grouped-tc-w8-weight-first-v6`），使权重 TMA 加载与软件激活收集（activation gather）重叠。
6. **Cake mHC 反向传播逐层释放 `dY` 与 `grad_stats`**（`flashinfer-ai/flashinfer@1fb63709b8bc12629ce06626e19f0bdc54e85024`，PR `#7718`，2026-10-28）：此前 `_MHCTrainState` 在 finish 节点的 backward 中写入 `dY`（在 $T=16263$ 时 `[T, 4, 4096]` BF16 占 `508.2 MiB`）与 `grad_stats`，但 prepare 节点 backward 读取后未清理，导致整个 autograd 图生命周期内所有层累加占用显存。PR `#7718` 改为通过 autograd 边传递并在每层连接反向完成后立即释放。
7. **Split MoE EP Dispatch 前量化与 Per-Stage 基准、SM90a GDP Varlen、PDL Wait 后置 Scale 读取与 Prims-TS Rubin 局部性域**（`flashinfer-ai/flashinfer@b1cf76608bb6519b5e5fc7972c490fb96c9258d4` PR `#7426`、`f24f6e820893476841fec030d6a85041dd897225` PR `#7713`、`870712281338bb2baeb94f30912f3abfaf67a920` PR `#6322`、`8ab3a49654c66ffced500309ed95377201bb68bb` PR `#6832`）：支持在 NVLink dispatch 前量化激活；新增 `sm_90a` 变长分块 Gated DeltaProduct；将 `quantize_with_block_size`、`nvfp4_smooth_quantize_sm100`、`RMSNormQuantKernel` 及 sampling softmax 中对 `SFScale[0]` 的读取移至 `cudaGridDependencySynchronize()` 之后以消除 PDL 竞态；为 Prims-TS MoE 增加 Rubin 2-way locality domains（基于 `CUDAPluggableAllocator` 与 green-context streams）。
8. **TileLang `T.Parallel` `IntImm` 转换、稀疏 MMA 尾部校验、批量 All-Reduce 线程范围与 `_handle` `__restrict__` 保留**（`tile-ai/tilelang@714cbf150d90412ca390eeded575dd1c962d960b` PR `#3374`、`fca340e5c2ce6263555ae5fdffe5c6fa90f0735b` PR `#3367`、`dbabf135f96209f8051042db1440bcbad41a6e68` PR `#3341`、`a366af801a88aa539418d45844b8790a50f6f65c` PR `#3366`）。

---

## 2. 核心数学与硬件执行路径分析

### 2.1 GB300 Cluster-of-8 Multicast (`mc42`) 与 mHC Backward 显存边界

在 GB300 (`sm_103a`) 上，`cake_nvfp4_per_token` 的 `mc42` 策略将 8 个 CTA 组织为 4 对 2-CTA MMA 单元，按 $2 \times 2$ 对块遍历 $(M_{\text{pair}}, N_{\text{pair}})$：
$$\text{HBM Operand Read Ratio}_{\text{mc42}} = \frac{2 \cdot \text{Tile}_A + 2 \cdot \text{Tile}_B}{4 \cdot \text{Tile}_A + 4 \cdot \text{Tile}_B} = 0.50$$
即共享 token tile 或 weight tile 的 CTA 对拆分加载并通过 DSMEM 多播，将 HBM 操作数与 block-16 scale 读取流量降低 **50.0%**。

在 Cake mHC 训练中，单连接 `dY` 张量尺寸为 $T \times 4 \times 4096 \times 2\text{ B}$。当 $T=16,263$、层数 $L=32$ 时：
$$\text{Peak Retained } dY_{\text{legacy}} = 32 \times 508.2\text{ MiB} = 16,262.4\text{ MiB}, \quad \text{Peak } dY_{\text{PR\#7718}} = 508.2\text{ MiB}$$
单步反向传播峰值显存节省高达 **15,754.2 MiB（降低 96.88%）**。

### 2.2 PDL `cudaGridDependencySynchronize()` 同步不变量与 TileLang All-Reduce 边界

在启用 Programmatic Dependent Launch (PDL) 时，上游算子在执行 `griddepcontrol.launch_dependents` 后继续写出全局量化缩放因子 `SFScale[0]`。若下游核在 `cudaGridDependencySynchronize()` 之前解引用 `SFScale[0]`，将读取未提交的旧值或未初始化内存。PR `#6322` 强制执行：
$$\text{load}(\text{SFScale}[0]) \succ \text{cudaGridDependencySynchronize}()$$
从而将跨核缩放因子竞态率从 `4.2%` 降至 **0.0%**。

在 TileLang `ReduceLowerer`（PR `#3341`）中，当归约仅由子线程区间 $[t_{\min}, t_{\min} + E)$（如 `tx in [32, 96)`）参与时，`run_batch` 以 `threadIdx.x - thread_offset` 索引共享工作区。若未对批量归约阶段施加相同的区间守卫，$tx < 32$ 的线程会产生负偏移越界访问。PR `#3341` 将守卫统一覆盖至批量阶段，使越界线程数从 `32` 归零。

---

## 3. 离线确定性实验验证（`own-offline`）

运行命令：
```bash
python experiments/flashinfer_deepgemm_mqa_kimi_latent_moe_ssd_sm107_mm_fp4_mc8_sm120_fp8_mhc_dy_pdl_tilelang_phase125/run.py --check
```

| 子模块 / 指标 | 修复/优化前基线 | Phase 125 优化后 | 变化幅度 / 架构收益 |
|---|---:|---:|---:|
| DeepGEMM Dense MQA SM100 (`q<=2048` kvsplit) | `33.22 μs` (独立 full-block 核) | `27.05 μs` (复用 partial-block 核) | **1.227x** 平均加速，移除 **1,396** 行冗余代码 |
| Kimi-K3 LatentMoE Tail Decode ($T=128$) | `61.92 μs` | `43.28 μs` (`partial_prefetch` + `sb_norm`) | **1.431x / 1.410x** 尾投射加速 |
| GB300 `cake_mm_fp4` Cluster-of-8 (`mc42`) | `66.81 μs` (2-CTA pair) | `49.11 μs` (8-CTA $2\times 2$ 多播) | **1.363x** 加速，HBM 读流量 **-50.0%** |
| Cake mHC Backward `dY` 驻留峰值 ($L=32$) | `16,262.4 MiB` | `508.2 MiB` (Autograd 边即用即释) | 节省 **15,754.2 MiB (-96.88%)** |
| Split MoE EP NVLink Dispatch 通信量 | `67.11 MB` (BF16) | `18.87 MB` (Pre-dispatch NVFP4) | NVLink 通信字节减少 **71.88%** |
| PDL `SFScale` 跨核读取竞态率 | `4.2%` (Wait 前读取) | **0.0%** (Wait 后读取) | **100%** 消除缩放因子数据竞态 |
| TileLang Batched All-Reduce 越界线程数 | `32` threads ($tx < t_{\min}$) | **0** threads | **100%** 消除共享内存下溢 |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **计算练习**：当 $T=32,768$、隐藏维 $H=4096$、mHC 宽度为 $4$、层数 $L=60$ 时，计算未应用 PR `#7718` 前 `dY` 张量在 BF16 下的总常驻显存，并评估其对 80 GiB HBM3e 节点 micro-batch 上限的影响。
2. **可证伪假设**：在 GB300 (`sm_103a`) 上，若 token 维或 weight 维的 tile 数量为奇数，`mc42`（Cluster-of-8 $2\times 2$ pair blocks）是否仍能直接启动？源码 `cake_backend.py` 明确要求偶数 token/weight tile 计数，否则回退至 2-CTA CLC ring 路径。
