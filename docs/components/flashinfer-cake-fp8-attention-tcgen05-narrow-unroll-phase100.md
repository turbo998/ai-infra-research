# FlashInfer Cake SM90 INT8/FP8 四阶段 Attention、`cake_tinygemm2_tcgen05` 窄分块展开、`cake_fp4_allreduce` 注册表收敛与 TileLang SIMT 向量局部偏移分析（Phase 100 · 100-Phase Milestone）

- **日期 / 阶段**：2026-10-27 · Phase 100 (`priority: 109` — **100-Phase Milestone!**)
- **组件归属**：`gpu-compute-architecture`
- **关联问题 ID**：`q-flashinfer-cake-fp8-attention-tcgen05-narrow-unroll-fp4-allreduce-phase100`
- **关联证据 ID**：`claim-flashinfer-cake-fp8-sm90-tcgen05-narrow-unroll-phase100`
- **可复现实验套件**：[`experiments/flashinfer_cake_fp8_tcgen05_phase100/`](../../experiments/flashinfer_cake_fp8_tcgen05_phase100/run.py)

---

## 1. 架构背景与核心问题

在 Hopper (`sm_90a`) 与 Blackwell (`sm_100a` / `sm_103a` / `sm_107a` / `sm_120a`) 混合精度算子栈演进中，工程落地面临四类核心挑战：

1. **长序列非因果 MHA 的动态平滑量化开销与端到端收益边界**：低比特 Attention（如 INT8 QK + FP8 PV）每步均需在 GPU 上动态计算统计量、执行 K-mean 平滑与量化。若统计与量化内核开销超过 Tensor Core 加速收益，短序列场景反而劣于 BF16 FlashAttention-3。
2. **Blackwell `tcgen05` 小 Batch 窄分块（Narrow Tile）的循环控制开销**：在 `flashinfer.gemm.tinygemm_bf16` 中，窄分块（Narrow Tile）过去按单轮（1-round）循环发射 MMA 与 TMA 加载，在 `sm_100a`/`sm_103a`/`sm_107a` 上受制于每步 K 边界谓词测试与循环分支开销。
3. **自动导出内核包（`cake_fp4_allreduce`）的编译膨胀与死代码淘汰**：`cake_fp4_allreduce` 包含 116 个 `.cu` 文件（252,517 行），其中大量预取（`_pf`）变体在真实 GPU 网格上限下永远不可达，且每个内核重复生成独立 TVM-FFI host binding。
4. **cuDNN/FA2 图捕获（CUDA Graph）生命周期安全与 TileLang Ascend PTO 向量局部寻址**：在 `auto` 后端切换或非单位步长空循环展开时，陈旧的 FA2 归并调度表或未乘以向量通道数（`element_lanes`）的局部偏移会导致静默精度损坏。

---

## 2. 一手源码实证分析（10 个零重叠上游提交）

### 2.1 FlashInfer `#7373` (`14165a5bd9`)：SM90 Cake INT8 QK + FP8 E4M3 PV 四内核链

`flashinfer/experimental/cake_fp8_attention` 通过 `prepare_fp8_attention(q, k, v, out, backend="cake")` 与 `fp8_attention_workspace_bytes` 为 `sm_90a` 提供 $D = 128$ 稠密非因果 MHA（BF16 输入/输出，FP32 Softmax，支持任意 NHD/HND 步幅视图）。单次调用在 CUDA Graph 重放下串行执行 **4 个内核**：

$$\text{BF16 } (Q,K,V) \xrightarrow{\text{1. stats\_partials}} \text{Partials} \xrightarrow{\text{2. stats\_finalize}} (\mu_K, s_Q, s_K, s_V) \xrightarrow{\text{3. quantize}} (\hat{Q}_{\text{INT8}}, \hat{K}_{\text{INT8}}, \hat{V}_{\text{E4M3}}) \xrightarrow{\text{4. attention}} \text{BF16 Out}$$

在 NVIDIA H100 80GB HBM3 (`sm_90a`, CUDA 13.4 JIT) 上对比完整 BF16 FlashAttention-3 调用（含统计与量化开销，冷 L2 中位数）：

| 形状编号 | Batch ($B$) | 序列长度 ($S$) | 头数 ($H$) | Cake 端到端耗时 (ms) | FlashAttention-3 BF16 (ms) | 端到端加速比 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `R1` | 1 | 4,800 | 40 | `0.7530` | `0.6484` | **`0.86x`** | 短序列下四内核统计/量化开销超过主核收益 |
| `R2` | 1 | 32,760 | 40 | `28.9556` | `31.6100` | **`1.09x`** | $O(S^2)$ 突破盈亏平衡点 |
| `R3` | 1 | 75,600 | 40 | `154.6865` | `176.3442` | **`1.14x`** | 超长序列 WGMMA INT8/FP8 吞吐主导 |
| `R4` | 1 | 32,760 | 5 | `3.7545` | `4.0651` | **`1.08x`** | 少头数长序列稳定正收益 |
| `R5` | 1 | 75,600 | 5 | `19.2451` | `22.0230` | **`1.14x`** | 少头数 75.6K 序列加速 14% |
| `R6` | 2 | 32,760 | 40 | `58.0676` | `63.9349` | **`1.10x`** | 双 Batch 32.7K 序列加速 10% |
| `R7` | 1 | 32,760 | 12 | `8.7903` | `10.2247` | **`1.16x`** | 峰值加速 **1.16x** |

### 2.2 FlashInfer `#7370` (`e177fd16c9`)：`cake_tinygemm2_tcgen05` 窄分块多轮展开与窗口重调

针对 Blackwell `tinygemm_bf16` (`out = input @ weight.T + bias`) 的 `cake_tinygemm2_tcgen05` 后端，PR `#7370` 将 Narrow Tile 的 MMA 与 TMA 加载循环按轮块（blocks of rounds）展开，并按架构微分派：
- **`sm_100a` (B200)**：采用 **2-round** 展开构建；在 `1x128x14336` 上带 Bias 耗时从 `7.58 µs` 降至 `6.85 µs`（**`1.11x`**），无 Bias 从 `7.42 µs` 降至 `6.88 µs`（**`1.08x`**）；`7x128x4096` 加速 **`1.04x–1.05x`**。
- **`sm_103a` (GB300)**：采用 **4-round** 展开构建，整阶段（full stages）直接发射而无需每步 K 边界分支测试（`without a per-step K test`）。
- **`sm_107a` (R200)**：采用 **4-round** 展开构建。
- 重测后每架构程序数从 **12 个扩展至 20 个**，更多长 $K$ 小 Batch 形状命中 Narrow Tile。

### 2.3 FlashInfer `#7366` (`5b79a1f222`)：`cake_fp4_allreduce` 死预取裁剪与共享 Binding

`cake_fp4_allreduce`（支撑 B200/GB300/R200 的 NVFP4 传输 + 融合 Residual + RMSNorm AllReduce）通过四步机械收敛大幅降低 JIT 体积：
1. **删除 5 个不可达融合归一化预取程序**：这些 `_pf` 程序仅在启动网格 $\ge 212$ CTAs 时才会被选中，但没有任何已出货 GPU 路由落入该区间（净删 21,615 行）。
2. **折叠 14 个 `_pf` 孪生内核**：通过编译宏 `-D` 将预取变体合入基础内核。
3. **53 个独立 Binding 合并为 10 个按形态共享的 Binding**：通过 `-DCAKE_FP4_ALLREDUCE_KERNEL`、`-DCAKE_FP4_ALLREDUCE_THREADS`、`-DCAKE_FP4_ALLREDUCE_DYNAMIC_SMEM` 参数化驱动。
4. **总体收敛**：`.cu` 文件从 **116 降至 49**（39 内核 + 10 共享 Binding），程序从 **58 降至 53**，JIT 模块从 **75 降至 68**，代码净减 **81,881 行**（`+2,038 / -83,919`）。

### 2.4 FlashInfer `#6231` (`04430a391b`)、ベンチマーク修复 `#7193`/`#7105`/`#7420` 与 TileLang `#3523`/`#3558`/`#2887`

- **cuDNN 预填计划复用与 Graph 模式 FA2 粘性保护 (`#6231`)**：当 CUDA Graph 先捕获 FA2、随后 `auto` 计划切向 cuDNN 时，若仅刷新 Q/KV `indptr` 而不刷新 FA2 的 CPU 生成输出归并调度表（output/merge schedule），会导致旧捕获图读取陈旧分块。修复后 Graph 模式一旦规划过 FA2 即保持 FA2 以持续刷新其调度表；同时在 `sm_100a`/`sm_107a` 上复用有界 batch/Q/KV 桶（`B1–4, Q2–1024, KV2K–32K`）。
- **基准与构建鲁棒性 (`#7193`, `#7105`, `#7420`)**：在 CUDA Graph 验证前刷新 wrapper 持有的输出缓冲（`#7193`）、修正 Frost FMA 路由测试使输入非零（`#7105`），以及清理 cubin 临时编译产物避免 `/tmp` 配额耗尽（`#7420`）。
- **TileLang Ascend PTO 向量局部寻址与空步长循环展开 (`#3523`, `#2887`, `#3558`)**：
  - `#3523` 引入 `SIMTVectorLocalScalarOffset(index, element_lanes)`，将向量槽位索引 $s$ 映射为标量偏移 $s \times L$（如 `float32x2`/`float32x4`），修复 SIMT VF 内向量局部缓冲越界与 AST 重写失败。
  - `#2887` 在 `LoopUnroller` 中当 `extent == 0` 时直接返回 `0`（而非对非单位 `step` 计算 `1 + (0 - 1) / step`），并按通道（lane-wise）下沉 ROCm/CUDA 向量化 `SelectNode`。

---

## 3. 定量实验验证（Python 3.12 标准库）

运行 `python experiments/flashinfer_cake_fp8_tcgen05_phase100/run.py` 生成确定性验证结果：

- Cake SM90 INT8/FP8 四阶段链在 `R2–R7` ($S \in [32760, 75600]$) 达到 **`1.08x–1.16x`** 加速（`R3` 有效吞吐 **946.07 TFLOP/s** vs FA3 BF16 **829.89 TFLOP/s**），而 `R1` ($S=4800$) 为 **`0.86x`**，验证了动态平滑量化的序列长度阈值。
- `cake_fp4_allreduce` 编译单元从 `116` 降至 `49`（减少 **57.76%**），净减 `81,881` 行生成代码。
- TileLang 空步长循环（`extent=0, step=4`）展开次数严格为 `0`，`float32x4` 槽位 `slot=3` 标量偏移精确映射为 `12`。
