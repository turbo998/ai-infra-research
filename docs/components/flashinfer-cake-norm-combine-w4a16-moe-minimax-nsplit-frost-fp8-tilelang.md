# FlashInfer CAKE Norm-Combine 打包归约、W4A16 拆分融合 MoE、MiniMax-H3 N-Split 尾波分块、Frost FP8 解码与 TileLang AST 源码映射

> English version: [flashinfer-cake-norm-combine-w4a16-moe-minimax-nsplit-frost-fp8-tilelang.en.md](flashinfer-cake-norm-combine-w4a16-moe-minimax-nsplit-frost-fp8-tilelang.en.md)

## 1. 背景与核心问题

在 Blackwell (`sm_100a` B200 / GB200、`sm_103a` B300 / GB300)、工作站 Blackwell (`SM120` RTX PRO 6000) 以及 Hopper (`SM90`) 上部署混合精度 MoE 与混合注意力模型时，算子工程面临五个紧耦合的瓶颈：

1. **多卡残差通信 + RMSNorm 归约延迟 (`cake_fused_norm_combine`)**：在 8 卡 NVLink 域上对 `BF16` (`H=2560`, 2 条 norm track) 执行 peer 归约与归一化时，若先等待远端控制块与 barrier 同步再发起本地 gamma/residual 流式加载，或逐标量执行 `BF16 -> FP32` 累加，小 batch (`T=1..64`) 受制于同步流水泡，大 batch (`T=1024..2048`) 受制于解包指令数。
2. **Blackwell `BF16 x NVFP4` (W4A16) 拆分融合 MoE 的批次切换边界 (`cake_fused_moe`)**：当专家权重以 `NVFP4` (E2M1 + UE4M3 block-16 scales) 存储而激活保持 `BF16` 时，CAKE 拆分式 gated SwiGLU FC1 / FC2 核函数在极小解码批次与超大吞吐批次占优，但在中等批次 (`T=128`) 会出现相对 CUTLASS 的交叉回退，架构师必须精确识别其调度分界。
3. **持久化分块量化 GEMM 的尾波浪费 (`minimax_h3_out_proj_nvfp4` & `mm_bf16_fp4`)**：当 MiniMax-H3 注意力输出投影的最后一个持久化 wave 仅占不到半个 wave 时，完整 `256 x 256` tile 会浪费一半以上的 CTA 周期。
4. **Hopper `SM90` 与 Blackwell `SM120` 低延迟 MoE 解码的 FP8 数值边界 (`Grouped Frost`)**：在 `T=4..8`、`top-k=10` 的 Grouped Frost 解码管线中，若沿整个 `K=128` TMA 分块直接累加 FP8 原生部分和，会在不放宽数值阈值的情况下破坏 BF16 激活前边界或 DeepSeek `128x128` 块缩放精度。
5. **长上下文 32 位偏移溢出与热路径主机发射开销 (`minimax_h3` SM120 varlen / cuDNN warm stream / TileLang AST)**：SM120 变长注意力以有符号 32 位偏移索引 `[padded_tokens, heads=56, 128]`，在约 `29.95 万` token 处发生有符号整型溢出；同时 Python 端每次调用 `torch.cuda.current_stream(device)` 构造包装对象及 DSL 宏展开丢失 AST 列范围都会增加运行时与调试成本。

---

## 2. 第一手源码实现路径审计

### 2.1 `cake_fused_norm_combine`：打包 `BF16x2` Peer 求和与前置流式加载 (`flashinfer@66a81f1bca98`, #7330)

在 `csrc/cake_fused_norm_combine/` 针对 `sm_100a` 和 `sm_103a` 的 4 档 token 计数的生成核函数 (`__launch_bounds__(320)`) 中：
- **前置流式加载 (Early Stream Loads)**：在 `griddepcontrol.wait` 之后，不再立即读取 `workspace[24]` 控制字并执行 `barrier.sync 1, 320`，而是先由 320 个线程完成 `gamma0[8]`、`gamma1[8]` 以及首个 token 块的 `x` 与 `residual` (`ld.global.v8.b32`) 预取，将全局内存延迟隐藏在跨卡栅格同步之前。
- **打包 `BF16x2` Peer 求和与单趟流水主体**：对 8 个 peer 的对称缓冲载荷采用打包 `__nv_bfloat162` 累加与打包归一化转换，配合 3 槽环形缓冲区 (`control_current = word % 3`, `control_dirty = (word + 2) % 3`)，每 token 对应 `160` 个 `v8.b32` 向量。
- **实测端到端延迟 (`H=2560`, 8 Peers, 2 Norm Tracks)**：
  - **8×B200 (`sm_100a`)**：`T=1` 从 `10.112 us` 降至 `9.184 us` (`1.101x`)；`T=8` 从 `11.520 -> 10.528 us` (`1.094x`)；`T=64` 从 `14.175 -> 12.928 us` (`1.096x`)；`T=256` 从 `16.768 -> 15.776 us` (`1.063x`)；`T=1024` 从 `29.792 -> 28.416 us` (`1.048x`)；`T=2048` 从 `46.560 -> 42.881 us` (`1.086x`)，6 档几何平均加速 **`1.0812x`**。
  - **8×B300 (`sm_103a`)**：`T=1` 从 `10.272 us` 降至 `9.280 us` (`1.107x`)；`T=8` 从 `12.384 -> 11.392 us` (`1.087x`)；`T=64` 从 `15.008 -> 13.792 us` (`1.088x`)；`T=256` 从 `17.472 -> 16.576 us` (`1.054x`)；`T=1024` 从 `30.688 -> 28.991 us` (`1.059x`)；`T=2048` 从 `47.008 -> 43.872 us` (`1.071x`)，6 档几何平均加速 **`1.0775x`**。

### 2.2 `cake_fused_moe`：`BF16 x NVFP4` W4A16 拆分融合 MoE 与中批次反证边界 (`flashinfer@cf5c9a6a415b`, #7315)

`flashinfer` 在 `cutlass_fused_moe(backend="cake")` 下新增了针对 `sm_100a` 和 `sm_103a` 各 10 个 program 的 `BF16` 激活 × `NVFP4` 权重 (W4A16) 拆分融合 MoE 核函数，包含 12 字节紧凑 `CakeFastDivmod` 结构体与独立的 gated SwiGLU FC1 / FC2 阶段：
- **B300 (`sm_103a`) 实测结果与反证边界**：
  - **Preset D (`H=7168, I=2048, E=256, top-8, 1 GPU`)**：在单 token 解码 `D1 (T=1)` 下，CAKE 阶段 1/2 耗时 `50.98 / 51.40 us`，优于 CUTLASS 基线 `53.73 us` (**`1.054x`**)；但在中等批次 `D128 (T=128)` 下，CAKE 耗时 `1027.63 us`，慢于 CUTLASS 的 `974.94 us` (**`0.949x`**)。
  - **Preset G (`H=6144, I=2048, E=256, top-8, EP8`)**：在 `G8 (T=8)` 达 `55.98 vs 58.63 us` (**`1.047x`**)，`G32` (`1.007x`)、`G64` (`1.010x`)、`G512` (`225.70 vs 227.97 us`, `1.010x`) 及 `G4096` (`310.51 vs 317.56 us`, **`1.023x`**) 均胜出；唯独在 `G128 (T=128)` 为 `133.62 vs 123.31 us` (**`0.923x`**)。这证明拆分式 W4A16 核在 `T=128` 附近存在中间激活显存往返与 tile 填充率次优区，生产路由需在 `T \in [96, 256]` 区间切换回 CUTLASS Fused MoE。

### 2.3 `minimax_h3_out_proj_nvfp4` N-Split 尾波分块与 `mm_bf16_fp4` M128 Pair (`flashinfer@b4125442a533`, #7314; `2866b59272d7`, #7311)

- **N-Split 半宽尾块机制 (#7314)**：针对 `minimax_h3_out_proj_nvfp4`（E2M1 数据 + UE4M3 block-16 缩放 + 全局缩放，带融合索引门控与残差），当形状满足 $M \ge 32$、总 wave 数不超过上限（`SM103` $\le 48$ waves，`SM100` $\le 10$ waves）且最后一个不完整 wave 占用率 $\le 0.5$ wave 时，将尾部 tile 沿输出列 $N$ 拆分为 **`256 x 128` 半宽单元** 执行，使尾波开销减半且输出与主干 **逐比特完全一致 (bitwise identical)**。在 GB300 (`sm_103a`) 触发尾拆分的 20 个形状上取得 **`1.003x–1.041x`** 加速（几何平均 **`1.021x`**，例如 `P=8, M=4816` 从 `82.2 -> 79.0 us`, `1.041x`）；在 GB200 (`sm_100a`) 的 7 个形状上取得 **`1.017x–1.052x`** 加速（几何平均 **`1.042x`**）。
- **`mm_bf16_fp4` `tcgen05` M128 Pair PDL 核函数 (#7311)**：为 `768 x 2112 x 2048` 等形状引入 M128 2-CTA pair 协同与 PDL (Programmatic Dependent Launch)，在 GB300 上实现原生 **`1.015x`** / tiled **`1.029x`** 加速，在 B200 上实现 **`1.018x`** / **`1.023x`** 加速。

### 2.4 Grouped Frost FP8 / DeepSeek Block-FP8 低延迟解码 (`flashinfer@d3d7c2488783`, #7212)

在 `H=2048, I=512, top-k=10` (`T=4, 8`) 的 Grouped Frost 低延迟 MoE 解码后端中：
- **架构候选集扩展**：
  - **Hopper `SM90` per-tensor FP8 → BF16**：采用 `WGMMA group2/stages4/K128`（`T=4, 8` 支持 `SwiGLU`/`SiTU`，`T=8` 支持 `ReLU2`）。
  - **Blackwell `SM120` per-tensor FP8 → BF16**：采用原生 warp FP8 MMA `group2` 或 `group4`、`stages2/K128`（`T=4, 8` 支持 `SwiGLU`/`SiTU`/`ReLU2`）。
  - **`SM90`/`SM120` DeepSeek `128x128` Block-FP8 → BF16**：复用 `128x128` 权重块缩放与动态 `128` 元素激活量化（`T=4, 8` 支持 `SwiGLU`，`T=8` 支持 `ReLU2`/`SiLU`）。
- **`K64` 原生部分和与 FP32 提升不变量**：尽管 TMA 分块沿 $K$ 维跨度为 `K128`，FP8 Tensor Core 累加被严格限制在最多跨越 **`K64`** 的原生部分和内，随后通过独立的 **FP32 加法** 提升（每个 `K128` TMA tile 发生 $128 / 64 = 2$ 次 FP32 提升），从而在不放宽数值阈值的前提下保持 BF16 激活前边界与静态/动态中间量化精度。

### 2.5 SM120 变长 32 位偏移保护、cuDNN 热流查找与 TileLang AST Span (`flashinfer@a02c8c5e6b57`, #7320; `85fcf535775b`, #7331; `tilelang@8f4501f2c3fd`, #3558)

- **SM120 `MiniMaxH3VarlenPlan` 32 位偏移边界 (#7320)**：SM120 核函数使用有符号 32 位整数 (`_MAX_ELEMENTS = (1 << 31) - 1 = 2,147,483,647`) 索引三个张量：
  $$\text{padded\_tokens} \times H \times D \le 2^{31}-1, \quad \max(S \times D,\; N_{\text{tiles}} \times 2D) \times H \le 2^{31}-1$$
  对于 `heads = 56, head_dim = 128, _BLOCK_N = 128`，单行元素数为 $56 \times 128 = 7168$，最大安全对齐 token 数为：
  $$\left\lfloor \frac{2147483647}{7168 \times 128} \right\rfloor \times 128 = 299{,}520\text{ tokens}$$
  一旦 `total_tokens = 299,521`（向上对齐至 `299,648`），偏移量达 $2{,}147{,}876{,}864 > 2^{31}-1$，计划器现在会立即抛出 `ValueError` 而非静默产生 GPU 越界读写。同理，空段统计上限为 `299,593` 段，单 token 查询分块统计上限为 `149,796` 个 query tiles。
- **cuDNN Warm Attention 原始流指针直查 (#7331)**：通过 `_get_raw_stream(device_index)` 直接读取底层 CUDA stream 句柄，绕过热路径上 `torch.cuda.current_stream(device)` 的 Python 对象分配，在 B200 上将 prefill 主机端发射延迟从 `32.796 us` 降至 `31.154 us` (**`-5.00%`**)，decode 从 `35.717 us` 降至 `32.706 us` (**`-8.43%`**)。
- **TileLang Eager AST 源码位置保留 (#3558)**：`QuoteVisitor` 将模板脚手架折叠到语句首行 `(lineno, col_offset, lineno, col_offset)`，但对替换进来的用户 AST 节点完整保留 `(lineno, col_offset, end_lineno, end_col_offset)` 四元组，并在 `visit_For` 中将 `__tb.iter_call` 的位置复制到用户迭代器调用表达式上，确保内联函数与循环展开的报错精确定位到用户表达式。

---

## 3. 可复现 CPU/Offline 验证实验与量化汇总

运行以下命令可复现本节全部数值模型与不变量校验：

```bash
python experiments/flashinfer_cake_norm_combine_w4a16_moe_minimax_nsplit_frost_fp8_tilelang/run.py --check
python -m unittest tests/test_phase75_experiments.py
```

| 子模块 / 算子 | 硬件 / 架构 | 关键配置 / 形状 | 优化前 / 基线 | 优化后 / 新方案 | 加速比 / 核心指标 |
|---|---|---|---|---|---|
| `cake_fused_norm_combine` (#7330) | 8×B200 (`sm_100a`) | `BF16, H=2560, T=1..2048` | `10.112..46.560 us` | `9.184..42.881 us` | 峰值 **`1.101x`** (`T=1`)，几何平均 **`1.0812x`** |
| `cake_fused_norm_combine` (#7330) | 8×B300 (`sm_103a`) | `BF16, H=2560, T=1..2048` | `10.272..47.008 us` | `9.280..43.872 us` | 峰值 **`1.107x`** (`T=1`)，几何平均 **`1.0775x`** |
| `cake_fused_moe` W4A16 (#7315) | B300 (`sm_103a`) | Preset D (`T=1`) / Preset G (`T=8, 4096`) | `53.73 / 58.63 / 317.56 us` | `50.98 / 55.98 / 310.51 us` | **`1.054x` / `1.047x` / `1.023x`** (`T=128` 为 `0.949x/0.923x`) |
| `minimax_h3_out_proj_nvfp4` (#7314) | GB300 (`sm_103a`) / GB200 (`sm_100a`) | 20 shapes / 7 shapes (`256x128` N-split tail) | Full `256x256` tail wave | Half-width `256x128` tail | GB300 **`1.021x`** (up to `1.041x`) / GB200 **`1.042x`** (up to `1.052x`) |
| Grouped Frost FP8 Decode (#7212) | `SM90` & `SM120` | `H=2048, I=512, top-k=10, T=4..8` | BF16 / Naive FP8 | `K64` partial + FP32 promote | 每个 `K128` TMA tile 精确 **2 次** FP32 提升 |
| SM120 Varlen Offset & cuDNN Stream (#7320, #7331) | `SM120` / B200 | `heads=56, D=128` / Warm Attention | Silent overflow / `35.717 us` | Guard @ `299,520` / `32.706 us` | 零静默越界；Decode 主机发射 **`-8.43%`** |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **架构设计练习（W4A16 混合路由表）**：在 B300 (`sm_103a`) EP8 集群上部署 `H=6144, I=2048, E=256, top-8` 的 W4A16 MoE 服务时，请根据 `#7315` 的实测曲线设计一个基于当前微批次 token 数 $T$ 的双后端分发策略（在 $T \le 64$ 与 $T \ge 512$ 走 `cutlass_fused_moe(backend="cake")`，在 $T \in [96, 256]$ 走标准 CUTLASS），并计算当线上请求满足双峰分布（70% $T=8$ 解码，30% $T=128$ 混合块预填充）时相比单一后端的端到端节拍收益。
2. **可证伪后续问题**：若将 `minimax_h3` SM120 变长统计核中的 `num_tiles * _PARTIAL_FLOATS * num_heads` 索引提升为 `int64_t` 指针算术，在 `M > 300,000` 超长上下文下需要额外占用多少寄存器与地址生成单元 (AGU) 周期？
