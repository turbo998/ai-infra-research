# FlashInfer SM103 `mhc_fused_post_pre` 边界融合、`kimi_k3_tp12_tail` 协作启动、Blackwell 10/12-CTA 簇缓存 Softmax、SM103a `K=96` NVFP4 SVDQuant MMA 与 TileLang 负索引越界拦截（Phase 105）

> **Language / 语言**: **中文 (Chinese)** | [English (英文版)](flashinfer-mhc-post-pre-tp12-tail-softmax-svdquant-cutlass-tilelang-phase105.en.md)

## 1. 架构背景与核心问题

在 `GLM-5.3-Flash`（45 层，每层 Attention 与 MoE 之间存在 4 条流形约束超连接残差流 `[T, 4, 4096]` BF16，前向共 **89 个 `post -> pre` 边界**）、`Kimi-K3` TP12 尾归约以及 Blackwell 大词表采样与 `SVDQuant` 低秩补偿 GEMM 中，传统的分步算子面临四大架构瓶颈：
1. **mHC 边界三次 HBM 往返与微网格启动开销**：未融合路径依次调用 `mhc_post`、TF32 预归一化投影 GEMM（`[24, 16384]`）与 `mhc_pre_big_fuse_with_prenorm`（含 20 步 Sinkhorn 双随机归一化与 RMSNorm），将 `[T, 4, 4096]` 的 `residual_cur` 写回 HBM 1 次并重读 2 次（共 3 遍扫描），在 `T=11,476` 时产生 **`1.69 GB`** HBM 流量。
2. **单次启动跨 CTA 自旋等待的 Green-Context 死锁隐患**：`kimi_k3_tp12_tail` 在 `M <= 8` 下使用单次启动的 `k123` 核（低位尾 CTA 收集各 Rank Owner 列，Owner 归约轮询尾 CTA 散射槽位）。若采用普通启动且运行在 SM 数不足以同时驻留全网格的 Green-Context 分区上，已驻留 CTA 将无限自旋等待未调度 CTA。
3. **Blackwell 词表 Softmax 簇维度空洞与 SM103a NVFP4 SVDQuant K=64 切分碎片**：此前缓存簇 Softmax 仅支持 `1/2/4/8/16` CTA 簇，缺少 `10-CTA` 与 `12-CTA` 单波次簇；同时 SM103a 上 `mm_nvfp4_svdquant` 在 256-wide K 阶段固定发出 4 条 `K=64` MMA，未能利用 `K=96` 块缩放指令。
4. **CUTLASS CuTeDSL 版本门控退化与 TileLang 负常量索引越界**：CuTeDSL `blockwise_gemm.py` 仅对字符串 `"4.6"` 开启 `opt_level=2`，导致 `4.7/4.8` 退化；TileLang `LegalizeNegativeIndex` 仅将负索引加上一次 `extent` 而未检查 `index + extent < 0`，在更小负常量索引下生成缓冲区起始地址之前的越界访存。

## 2. 第一手源码实现与数学不变量

### 2.1 FlashInfer SM103 `mhc_fused_post_pre` 三合一边界融合（`f56fecc7ebf7989a5f075031ff0ded26000b1e8b`）
- **源码路径**：`flashinfer/experimental/mhc_fused_post_pre/backend.py`、`flashinfer/experimental/mhc_fused_post_pre/README.md`。
- **融合机制**：在单次核调用内完成第 $i$ 块的 `post map`（$\text{residual\_cur}_j = \text{post\_mix}_j \cdot x + \sum_i \text{comb\_mix}_{i,j}\cdot \text{residual}_i$）、TF32 预归一化投影（$[24, 16384]$，FP32 累加）以及第 $i+1$ 块的 20 步有序 Sinkhorn 归一化与输入 RMSNorm（覆盖 $T \in [33, 16384]$）：
  - 每个边界核启动从 `3 -> 1`（全模型 89 个边界从 **`267 -> 89` 次启动，每次前向省去 `178` 次启动**）；
  - `T = 11,476` 时 HBM 流量从 `1.69 GB` 压降至单遍理论下限 **`0.94 GB`（削减 `44.38%`，`1.7979x`）**；
  - GB300 实测：Decode `T=48` 从 `14.88 -> 6.88 us`（**`2.1628x`**），Mixed `T=11,476` 从 `310.96 -> 186.27 us`（**`1.6694x`**）。

### 2.2 `cake_kimi_k3_tp12_tail` 协作启动（`9a27746f1db861700fafe7579031aa5442a54184`）与 10/12-CTA Softmax（`e61cdfa925127e68f62781d3e81cf511f808c257`）
- **协作启动护栏**：将 `M <= 8` 的 4-token 与 8-token `k123` 核切换为 `cudaLaunchCooperativeKernel`，强制全网格同时驻留；若 Green-Context 分区过小则立即抛出确定性 `RuntimeError`（死锁从 **`1 -> 0`**，GB300 `M=1` 延迟 `17.59 -> 17.30 us` `1.0168x`）。
- **10/12-CTA 缓存簇 Softmax**：在 `csrc/cake_blackwell_softmax_cached_cluster.cu` 中新增 6 个单波次 `10-CTA`（`8/10/13` 寄存器瓦片）与 `12-CTA`（`8/10/11` 寄存器瓦片）程序（总数 `14 -> 20`），B200 在 `37 x 128000` 上从 `9.95 -> 8.51 us`（**`1.1692x`**），GB300 在 `84 x 100000` 上从 `14.43 -> 12.58 us`（**`1.1471x`**）。

### 2.3 SM103a `cake_nvfp4_svdquant` K=96 MMA（`4f3b2d6b91e26066d978a988e37b18c575cd305d`）与测试加固（`c76ffef227123a8a7c25af6165bb6d935c9a395f`）
- **K=96 块缩放 MMA 与 Split-K 11 级别名环形缓冲**：在 SM103a 的 18 个大形状上以 `96 + 96 + 64` 组合替代 4 次 `K=64` 的 `kind::mxf4nvf4.block16` MMA（每 256-K 阶段 MMA 指令数 **`4 -> 3`**），并在 `M=129, K=12288` 上共享 11 级操作数环与 256B L2 提升；同时在 `CDiv` 中拦截 `b == 0 || (a == INT64_MIN && b == -1)` 64 位有符号整数溢出。GB300 在 `(16384, 3072, 12288, 32, 1)` 上从 `162.46 -> 148.30 us`（**`1.0955x`**），GB200 在 `(129, 3072, 12288, 32, 0)` 上从 `11.01 -> 10.53 us`（**`1.0456x`**）。

### 2.4 CUTLASS CuTeDSL 优化门控（`147295a3d4`, `59e3a3338d`, `b87a77760f`）与 TileLang 负索引合法化检查（`004b7a2a69`, `fa456b8b2c`）
- **CUTLASS**：修复 `blockwise_gemm.py` 版本门控使 CUTLASS `>= 4.7` 启用 `opt_level=2`；将 `SM121` (`121`/`121a`) 默认编译目标移入 CUDA 12.9 分支；并在 `dsl.py` 中保持链接库确定性插入顺序。
- **TileLang**：`bench.py` 在未显式指定设备时使用 `torch.cuda.current_device()` 刷新 L2；`src/transform/legalize_negative_index.cc` 在 `index + extent < 0` 时直接拒绝越界负常量索引（测试集拦截 **`3` 个越界负索引**）。

## 3. 确定性实验与量化对比

运行 `python experiments/flashinfer_mhc_post_pre_tp12_tail_softmax_svdquant_phase105/run.py --check`：

| 指标 | 优化前基线 | Phase 105 实现 | 提升幅度 / 结果 | 证据分类 |
|---|---:|---:|---:|---|
| GLM-5.3-Flash 89 边界核启动次数 | `267` | `89` | **省去 `178` 次启动 (`-66.67%`)** | `implementation` + `own-offline` |
| `mhc_fused_post_pre` `T=11,476` HBM 流量 | `1.69 GB` | `0.94 GB` | **`-44.38%` (`1.7979x`)** | `author-benchmark` + `own-offline` |
| GB300 `mhc_fused_post_pre` Decode (`T=48`) / Mixed (`T=11,476`) | `14.88 us` / `310.96 us` | `6.88 us` / `186.27 us` | **`2.1628x` / `1.6694x`** | `author-benchmark` |
| Blackwell 缓存簇 Softmax (`37x128k` B200 / `84x100k` GB300) | `9.95 us` / `14.43 us` | `8.51 us` / `12.58 us` | **`1.1692x` / `1.1471x`** | `author-benchmark` |
| SM103a NVFP4 SVDQuant `(16384,3072,12288,32,1)` 延迟 | `162.46 us` | `148.30 us` | **`1.0955x` (`4 -> 3` MMAs/256K)** | `author-benchmark` |

## 4. 架构师设计练习与可证伪问题

1. **计算题**：当 `T = 4,096` 时，4 流 BF16 残差 `[T, 4, 4096]` 的单遍字节数为 $4096 \times 4 \times 4096 \times 2 = 134,217,728\text{ B} = 128\text{ MiB}$。求 89 个边界融合前后仅残差流在单步前向中节省的 HBM 读写总量（答案：$89 \times 2 \times 128\text{ MiB} = 22,784\text{ MiB} = 22.25\text{ GiB}$）。
2. **可证伪假设**：若在 Green-Context 仅分配 8 个 SM 的子分区上运行需要 12 个 CTA 跨块互等的 `kimi_k3_tp12_tail` (`M <= 8`)，非协作启动将 100% 挂起，而 `cudaLaunchCooperativeKernel` 将在内核提交阶段同步返回 `cudaErrorCooperativeLaunchTooLarge` 并转化为 `RuntimeError`。
