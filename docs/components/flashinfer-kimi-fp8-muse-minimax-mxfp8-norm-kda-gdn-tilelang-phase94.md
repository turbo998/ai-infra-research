# FlashInfer Kimi-K3 FP8 算子瘦身、Muse Spark SM107a PDL Norm、MiniMax-H3 SM103/SM120 投影、MXFP8 224 线程量化、Fused Norm-Combine `min.s16x2` 轮询、KDA Prefill、SM8x 分块并行 GDN 与 TileLang Ascend 精确片上布局解析（Phase 94）

> **English Version**: [flashinfer-kimi-fp8-muse-minimax-mxfp8-norm-kda-gdn-tilelang-phase94.en.md](flashinfer-kimi-fp8-muse-minimax-mxfp8-norm-kda-gdn-tilelang-phase94.en.md)  
> **Evidence Class**: `implementation` (上游一手源码提交) + `author-benchmark` (作者 PR 实测) + `own-offline` (仓库内 Python 3.12 标准库确定性复现)  
> **Experiment Path**: `experiments/flashinfer_kimi_fp8_muse_minimax_mxfp8_norm_kda_gdn_tilelang_phase94/`

---

## 1. 架构背景与核心问题

在 Blackwell (`sm_100a`/`sm_103a`/`sm_120a`)、Rubin (`sm_107a`)、Ampere (`SM8x`) 以及昇腾 Ascend NPU 的最新编译与算子工程中，Phase 94 Track 1 聚焦十项具有直接生产价值的微架构改造：

1. **Kimi-K3 FP8 投影 Launcher 瘦身与 Decode 分桶扩展（FlashInfer `#7399`）**：原本 54 个 Launcher 编译单元按参数签名合并为 5 个（`-90.74%`），5 对仅常量开关不同的双生 Kernel 折叠进 `#if` 分支（`75 -> 70`），移除不可达的 `_sks` Stream-K 分支，并将 `DECODE_TABLE` 分桶从 `M <= 256` 扩展至 `512/1024/2048`。
2. **Muse Spark Norm 的 Rubin `sm_107a` PDL 路由与带宽分档重调（FlashInfer `#7400`）**：在 Rubin R200（`sm_107a`，212 SMs）上为 `rmsnorm_sigmoid_gate` 引入 `bf16_pdl`（Programmatic Dependent Launch），并重调 `128/32/16` 头数的带宽分档阈值（`384 / 768 / 2048` tokens）。
3. **MiniMax-H3 SM103 256-bit 残差加载与 SM120 FP8/NVFP4 Pre-Attention 提速（FlashInfer `#7397`, `#7347`）**：在 `sm_103a`（GB300，152 SMs = 76 簇对）上将每 16 列 BF16 残差从 2 次 128-bit 合并为 1 次 256-bit 向量加载，并在 `griddepcontrol.wait` 前预发首个权重分片加载；在 `sm_120a`（RTX 5090 / RTX PRO 6000 Blackwell）上将 FP8->BF16 Pre-Attention 提速 `1.3655x`（`M=33472` 时 `18.6993 -> 13.6945 ms`）。
4. **Grouped MXFP8 224 线程行分块量化（FlashInfer `#7391`）**：针对 `K = 7168 = 224 x 32` 且 `B*M >= 16384` 的 MoE 路由输出，将每行从 2 个 128 线程 CTA 合并为 1 个 224 线程（7 warps）CTA，使 `[32, 1024, 7168]` 稀疏掩码下的 mask-blind Grid 减半（`65,536 -> 32,768` CTAs），实测提速 `1.41x–1.60x`。
5. **Fused Norm-Combine 打包 `min.s16x2` 就绪树与 4-CTA 控制记录预取（FlashInfer `#7390`）**：用 3 条 `min.s16x2` 指令替代 7 次标量比较轮询 8 个对称张量并行 peer 的 epoch 就绪号，并在 4-CTA 流水线 kernel 的 `griddepcontrol.wait` 后立即发起 `workspace[24]`/`workspace[16+rank]` 与首 token 异步拷贝（B200 上 1 token 延迟 `8.128 -> 7.712 us`，`1.054x`）。
6. **BF16 输入 + FP32 状态 KDA Prefill 重构与 SM8x 4 阶段分块并行 GDN（FlashInfer `#7388`, `#5109`）**：将 22 个 Planner 选中的 KDA 模块重构为每架构 36 个程序（GB300 `T=8192` 从 `275.78 -> 238.53 us`，`1.16x`；`BS4xT64` 达 `1.22x`）；在 SM8x 上实现 `T -> MN -> fixup -> prefill` 4 阶段分块并行 Gated Delta Rule，将 `8192` token 的串行递推链从 `128` 步压缩至 `4` 步（`32x` 串行深度缩减），并为无 `stmatrix` 的 pre-SM90 提供 `CopyR2SOp` 回退。
7. **TileLang Ascend 精确片上 Buffer 字节寻址与 `Int64Promoter` 越界防护（TileLang `#3607`, `#3625`）**：支持 `tl.ascend_buffer_addresses` 对齐与容量校验（`UB 248 KiB / SIMT 保留后 216 KiB`、`L1 512 KiB`、`L0A/B 64 KiB`、`L0C 256 KiB`、`SS 3 KiB`），并在 `RegionsMayConflict` 中将半开区间 `[min, min + extent)` 提升至 `int64`，彻底消除 `(1 << 31) - 1` 边界处的 `int32` 符号溢出导致的同步丢失。

---

## 2. 核心数学与硬件执行模型

### 2.1 Grouped MXFP8 224 线程行内单 CTA 量化（FlashInfer `#7391`）

对于 DeepSeek-V3 / Kimi 规模的隐藏维 $K = 7168$，MXFP8 采用 32 元素共享 1 个 `UE8M0` 缩放因子：
$$N_{\text{blocks/row}} = \frac{K}{32} = \frac{7168}{32} = 224$$
旧版 128 线程程序每 CTA 仅处理 128 个缩放块，因此每行需要 $\lceil 224 / 128 \rceil = 2$ 个 CTA。在 `[B=32, M=1024, K=7168]` 的分组稀疏激活下，即使 `masked_m` 仅激活少量行，Grid 仍按全量 $B \times M = 32,768$ 行发射：
- **128 线程双 CTA/行**：发射 $32,768 \times 2 = 65,536$ 个 CTA；
- **224 线程单 CTA/行（7 Warps）**：发射 $32,768 \times 1 = 32,768$ 个 CTA，空转 CTA 调度开销严格减半（`2.0x` 缩减），在 `count=16..256` 稀疏区间带来 `1.41x–1.60x` 加速。

### 2.2 SM8x 分块并行 Gated Delta Rule 与 TileLang `Int64Promoter`

SM8x 分块并行 GDN（FlashInfer `#5109`）将长度 $L = 8192$ 的序列划分为 $C = 2048$ 的大块与 $B_T = 64$ 的子块：
1. **Stage `T`（全并行）**：计算严格下三角逆矩阵 $(I + \text{tril}(\text{diag}(\beta) K K^\top, -1))^{-1}$；
2. **Stage `MN`（按 Chunk 并行）**：在每个 `2048`-token Chunk 内累积局部状态；
3. **Stage `fixup`（跨 Chunk 串行）**：仅在 $L / C = 8192 / 2048 = 4$ 个 Chunk 间传播跨块状态，相比逐 `64`-token 块串行的 $8192 / 64 = 128$ 步缩减了 $32\times$ 串行步数；
4. **Stage `prefill`（全并行）**：结合修正后状态并行生成最终输出。

在昇腾 NPU 编译器的自动同步插入（TileLang `#3625`）中，当缓冲区偏移贴近 `int32` 上界 $M_{32} = 2^{31} - 1 = 2,147,483,647$ 且 `extent = 1` 时，`int32` 算术下 `min + extent` 溢出为 $-2,147,483,648 < \text{min}$，破坏区间相交判定；`Int64Promoter` 在构造半开区间前递归将 `min` 与 `extent` 提升为 `int64`（$2,147,483,648 > 2,147,483,647$），并保留 `min/max` 尾部裁剪表达式，确保 `PIPE_MTE3 -> PIPE_MTE2` 屏障不漏插。

---

## 3. 确定性实验复现结果（`own-offline`）

运行 `python experiments/flashinfer_kimi_fp8_muse_minimax_mxfp8_norm_kda_gdn_tilelang_phase94/run.py --check` 通过全部断言：

| 模块与 PR | 关键指标 | 优化前 | 优化后 | 幅度 / 结论 |
|---|---|---:|---:|---|
| FlashInfer `#7399` Kimi-K3 FP8 | Launcher 源文件数 / Decode 最大桶 | `54` / `M<=256` | `5` / `M<=2048` | `-90.74%` 编译单元 |
| FlashInfer `#7397` / `#7347` MiniMax-H3 | SM103 16 列残差加载 / SM120 FP8 Pre-Attn | `2x128b` / `18.6993 ms` | `1x256b` / `13.6945 ms` | `1.3655x` (`M=33472`) |
| FlashInfer `#7391` Grouped MXFP8 | `[32, 1024, 7168]` Grid CTA 数量 | `65,536` (`128` 线程) | `32,768` (`224` 线程) | `2.0x` CTA 缩减 / `1.60x` 提速 |
| FlashInfer `#7390` Fused Norm-Combine | B200 `T=1` / `T=8` 延迟 (`H=2560`) | `8.128` / `10.560 us` | `7.712` / `10.048 us` | `1.054x` / `1.051x` |
| FlashInfer `#7388` / `#5109` KDA & GDN | GB300 KDA `T=8192` / GDN 串行步数 | `275.78 us` / `128` 步 | `238.53 us` / `4` 步 | `1.16x` / `32x` 串行链缩短 |
| TileLang `#3607` / `#3625` Ascend | SIMT 开启时 UB 可用容量 / `int64` 上界 | `248 KiB` (未扣) / 溢出 | `216 KiB` / `2147483648` | 精确容量与零溢出同步 |
