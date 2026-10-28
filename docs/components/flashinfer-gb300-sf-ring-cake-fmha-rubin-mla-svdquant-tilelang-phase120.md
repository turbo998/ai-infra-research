# FlashInfer GB300 独立 SF 环路 FP4 GEMM、CAKE 27 路分页 Decode 注意力、Rubin MLA 延后 PDL 触发与双模合并、SVDQuant 单波次尾块重组、MiniMax H3 与 Warp Decode、B12X MoE 2 幂次工作区桶化、cuDNN 1.32 QSA 与 TileLang ROCm 10 架构解析（Phase 120）

> **模块定位**：面向具有云基础设施背景、希望深入掌握 Blackwell Ultra (`sm_103a`) / Rubin (`sm_107a`) 算子微架构的解决方案架构师（Solution Architect）。本文档基于 `flashinfer-ai/flashinfer` 与 `tile-ai/tilelang` 在 2026-10-27 至 2026-10-28 合入的 10 个核心提交，系统拆解独立缩放因子（Scale-Factor）TMA 流水线环路、分页注意力 AOT 内核族、尾波次（Tail-Wave）不对称分块重组与编译器缓存防抖机制。
> **证据等级**：`implementation`（源码固定 commit 级核验）+ `author-benchmark`（上游 PR 实测数据，非本仓库 GPU 复测）+ `own-offline`（本仓库 Python 标准库可复现建模验证）。

---

## 1. 核心架构问题与 10 项上游证据总览

在 Blackwell (`sm_100a`, 148/152 SMs)、Blackwell Ultra (`sm_103a`, GB300 152 SMs) 与 Rubin (`sm_107a`, 208 SMs) 架构上，FP4/FP8 Tensor Core 的峰值吞吐极高，导致算子瓶颈从主矩阵乘（MMA）迅速转移到：
1. **块缩放因子（Block Scale Factor, SF）与主操作数（A/B）共享 TMA 流水线时的气泡与尾声（Epilogue）共享内存往返开销**；
2. **非整数波次（Tail Wave）在 76 个 2-CTA 对（152 SMs）上的低占用率**；
3. **动态路由 MoE 每次因 `routed_rows` 微小波动触发 `cuda.tile` / CuteDSL 全量重编译**。

| 序号 | 仓库与 Commit | 日期 | 核心机制与架构变更 | 证据分类 |
| :--- | :--- | :--- | :--- | :--- |
| A1 | `flashinfer-ai/flashinfer` [`4ca4e6e`](https://github.com/flashinfer-ai/flashinfer/commit/4ca4e6e485838462334de5aa9012f3895354341f) | 2026-10-28 | `cake_mm_fp4`：GB300 (`sm_103a`) 6 级 A/B 环路 + 独立 5 级 SF 环路与寄存器直写尾声 (`gemm_2cta_ptr`) | `implementation` + `author-benchmark` |
| A2 | `flashinfer-ai/flashinfer` [`5e0e99b`](https://github.com/flashinfer-ai/flashinfer/commit/5e0e99bd2e8be8f3ebdb3a3a0bd670ab114c425f) | 2026-10-28 | `cake_fmha`：新增 27 个分页 Decode 注意力 AOT 内核（BF16/FP8/NVFP4 KV/head-dim 256） | `implementation` + `author-benchmark` |
| A3 | `flashinfer-ai/flashinfer` [`f6587bc`](https://github.com/flashinfer-ai/flashinfer/commit/f6587bc5dd490e1b08a18040ab5fb6496b599603) | 2026-10-28 | `cake_rubin_mla_nvfp4_decode`：Rubin `sm_107a` 延后 PDL 触发 (`items <= units + pairs`) 与 `probe_after_loads` / `column` 合并 | `implementation` + `author-benchmark` |
| A4 | `flashinfer-ai/flashinfer` [`03ebbc6`](https://github.com/flashinfer-ai/flashinfer/commit/03ebbc6e7121749d802991553b4093383431c154) | 2026-10-28 | `cake_nvfp4_svdquant`：`sm_100a` `(4096, 3072, 12288, rank=32)` 40-Tile 尾波重排为单波次 76 单元 (`8×192 + 68×128`) | `implementation` + `author-benchmark` |
| A5 | `flashinfer-ai/flashinfer` [`d949946`](https://github.com/flashinfer-ai/flashinfer/commit/d949946f67275c61ab977a7806c9f6e806469da6) | 2026-10-28 | `cake_minimax_h3_varlen_attention`：双槽位 Ticket Mailbox（words 6 & 7）、`quantize_early_qk` 与 Tiny BF16 计划 | `implementation` + `author-benchmark` |
| A6 | `flashinfer-ai/flashinfer` [`7de971a`](https://github.com/flashinfer-ai/flashinfer/commit/7de971a26c89340e12d4298f9dfbd4a48da28e50) | 2026-10-27 | `cake_warp_decode`：SM100/SM103 Count-Rank Route-Pack 头部（129+10 行）与 Early-Load Finalizer（46+12 行） | `implementation` + `author-benchmark` |
| A7 | `flashinfer-ai/flashinfer` [`959a113`](https://github.com/flashinfer-ai/flashinfer/commit/959a113cefa2a8e43c924b4dc411f165164df2c5) | 2026-10-28 | `b12x`：静态 MoE 后端对 `routed_rows` 进行 2 的幂次桶化 (`1 << (max(128, R)-1).bit_length()`) 消除编译缓存抖动 | `implementation` |
| A8 | `flashinfer-ai/flashinfer` [`c2ba71f`](https://github.com/flashinfer-ai/flashinfer/commit/c2ba71f72c38f2ff29ec325dfcc8ba7f35743ef0) | 2026-10-28 | `cudnn`：集成 cuDNN 1.32 原生 QSA 前向 API（`qsa_forward` 位图模式与 `qsa_indexed_forward` 索引模式） | `implementation` |
| A9 | `tile-ai/tilelang` [`040b305`](https://github.com/tile-ai/tilelang/commit/040b3054f1c20b1514be0b8e497ccbb791009986) | 2026-10-27 | `target/utils.py`：ROCm `auto` 目标探测使用 `torch.cuda.current_device()` 替代硬编码设备 `0` | `implementation` |
| A10 | `tile-ai/tilelang` [`015caa2`](https://github.com/tile-ai/tilelang/commit/015caa2b6e351e91d3b54922b9d130a8ec3b99f0) | 2026-10-27 | `env.py` / `utils.py`：支持 pip-SDK ROCm 10（`rocm-sdk`）四级架构解析与 10 秒 `rocminfo` 超时回退 | `implementation` |

---

## 2. 深度技术剖析与公式推导

### 2.1 GB300 (`sm_103a`) 独立 Scale-Factor 环路与寄存器直写尾声（`4ca4e6e`）

在 `flashinfer/cake/kernels/mm_fp4/program.py` 中，针对 `SF_RING_ARCH = "sm_103a"`（GB300，152 SMs）：
- **双环解耦（Decoupled TMA Rings）**：传统 `sm_100a` K256 计划将 A/B FP4 数据块与 E4M3 缩放因子（SF）绑定在同一 4 级流水线环路中。GB300 拥有更大的共享内存容量，因此将主数据流水线扩展至 `NUM_TMA_PIPE_STAGES = 6`（6 级 K256 A/B 环路），同时为缩放因子单独开辟 `NUM_SF_PIPE_STAGES = 5` 的独立 `mbarrier` 环路。
- **寄存器直写尾声（Register-Direct Epilogue）**：在 `gemm_2cta_ptr` 路径中，TMEM 累加器读出到寄存器后直接转为 BF16 并写回全局内存，省去了“寄存器 $\to$ Shared Memory $\to$ TMA Store”的二次搬运与显式 fence。
- **上游实测收益（`author-benchmark`）**：
  - 在 16 个 DeepSeek/Llama FP4 GEMM 形状上取得 **1.035x–1.107x** 加速，其中 `16384×7168×2048` 从 `62.33 us` 降至 `56.35 us`（**1.1061x**）；
  - 在 14 个 `quant+GEMM` 链式基准上取得 **1.030x–1.084x** 加速，其中 `28672×8192×2048` 从 `156.53 us` 降至 `144.46 us`（**1.0836x**）。

### 2.2 CAKE FMHA 27 路分页 Decode 注意力与设备侧缩放消除同步（`5e0e99b`）

`cake_fmha` 为 `sm_100a` 和 `sm_103a` 新增了 **27 个 AOT 编译程序**（每个架构目标编译 19 个程序，覆盖 `D=128` 与 `D=256`、GQA 比例 `4/5/8/16`、KV Page Size `1/16/32/64` 以及 `BF16`、`FP8`、`NVFP4 KV`）：
- **Head-Dim 256 小 M 路径**：当 `Q_len * group_size <= 8` 时启用 `head_dim_256_small_m`，避免填充到 64 行的 MMA Tile 浪费算力；
- **设备侧 Scale 直读（Device-Scale Pointer）**：当 `q_scale` / `kv_scale` 驻留在 GPU 显存时，直接将设备指针传入内核，消除了原先隐式 `.item()` 触发的 Host-Device 同步阻塞。在 B200 FP8 `B=128, kv=4096` 上，延迟从 `59999.70 us` 骤降至 `169.89 us`（**353.17x**）；
- **B200 BF16 与 FP8 性能（`author-benchmark`）**：BF16 `B=4, kv=1024, gqa=8, D=128` 从 `13.44 us` 降至 `7.68 us`（**1.750x**）；FP8 `B=256, kv=8192` 达到 `317.31 us`（优于 `trtllm-gen` 的 `321.45 us`）。

### 2.3 Rubin (`sm_107a`) MLA NVFP4 Decode 延后 PDL 与 SVDQuant 单波次尾块重组（`f6587bc`, `03ebbc6`）

1. **Rubin `sm_107a` MLA NVFP4 Decode（`f6587bc`）**：
   - 将 Programmatic Dependent Launch (PDL) 触发条件从 `items <= units` 放宽至：
     $$\text{items} \le \text{units} + \text{pairs}$$
     在 208-SM Rubin 板卡上（`units = 50`, `pairs = 4`），阈值从 `50` 提升至 `54`，使 `51–54` 个工作项的形状也能提前拉起 Combine 内核；
   - 新增 `probe_after_loads`（先发射全部 Split 投机加载再探测完成状态）与 `column`（4-Warp，处理 `384–448` 行、`17–32` Splits，每线程每 Split 仅加载 2 个 BF16 字即 `8 Bytes`）两种归约形态，在 `H192-Q1-B2-KV128K` 上实现 **1.067x** 加速。
2. **SVDQuant 单波次尾块重组（`03ebbc6`）**：
   - 针对 `M=4096, N=3072, K=12288, rank=32` 在 152 SMs（76 个 2-CTA Pair）上的执行：标准 `256×256` Tile 共有 $(4096/256) \times (3072/256) = 16 \times 12 = 192$ 个 Tile。前两波次处理 $2 \times 76 = 152$ 个 Tile，剩余 **40 个 Tile**（共 $40 \times 256 = 10240$ 列×行切片）若按原粒度执行，将有 $76 - 40 = 36$ 个 CTA Pair 空转；
   - 新方案将 40 个尾块精确切分为 **76 个工作单元**：8 个 `192` 列分片（来自 `2×3` 块拆分）与 68 个 `128` 列半块：
     $$8 \times 192 + 68 \times 128 = 1536 + 8704 = 10240 = 40 \times 256, \quad 8 + 68 = 76\text{ CTA pairs}$$
     从而将两波半（3 Waves）压缩为严格的 **3 个满载单波次**，在 GB200 上将延迟从 `62.98 us` 降至 `59.17 us`（**1.0644x**）。

### 2.4 B12X MoE 工作区桶化、cuDNN 1.32 QSA 与 TileLang ROCm 10（`959a113`, `c2ba71f`, `040b305`, `015caa2`）

- **B12X MoE 2 幂次桶化（`959a113`）**：`cuda.tile` 将静态工作区第一维 `routed_rows` 烘焙进编译后 cubin。提交 `959a113` 引入：
  $$\text{bucket}(R) = 2^{\lceil \log_2(\max(128, R)) \rceil} = \text{1 << (max(128, int(R)) - 1).bit\_length()}$$
  当桶化后总元素数超过 32 位索引安全上限（$2^{31}-1$）时自动回退至精确 `routed_rows`。在 `64..4096`（步长 17，共 238 个不同批次大小）的扫描中，JIT 编译次数从 **238 次降至 6 次**（减少 **39.67x**）。
- **cuDNN 1.32 QSA 前向 API（`c2ba71f`）**：封装 `qsa_forward`（每 `uint8` 编码 8 个 `block_size=4` 的块共 32 个 Token，以 `s_q <= 512` 与 `s_q >= 513` 划分短/长序列内核）与 `qsa_indexed_forward`（`int32` Top-K 索引模式）。
- **TileLang ROCm 多卡与 pip-SDK ROCm 10（`040b305`, `015caa2`）**：修复异构多卡下硬编码 `get_device_properties(0)` 的 Bug，改为查询 `torch.cuda.current_device()`；并为 pip 安装的 ROCm 10（`rocm-sdk`）建立四级架构探测链（显式 `rocm_path` $\to$ 10 秒限时 `rocminfo` $\to$ PyTorch `gcnArchName` $\to$ `"gfx900"`）。

---

## 3. 本仓库离线建模验证（`own-offline`）

对应实验目录：`experiments/flashinfer_gb300_sf_ring_cake_fmha_rubin_mla_svdquant_tilelang_phase120/`

- **运行命令**：`python experiments/flashinfer_gb300_sf_ring_cake_fmha_rubin_mla_svdquant_tilelang_phase120/run.py --check`
- **关键输出校验**：
  - `gb300_sf_ring_fp4_gemm.max_gemm_speedup = 1.1061`，`max_chain_speedup = 1.0836`；
  - `cake_fmha_paged_decode.bf16_b4_kv1024_speedup = 1.75`，`device_scale_speedup = 353.17`；
  - `rubin_mla_and_svdquant.svdquant_single_wave_fit = true`（`8*192 + 68*128 = 10240` cols, `76` units）；
  - `b12x_qsa_tilelang_rocm.b12x_compile_reduction_ratio = 39.67`（238 形状收敛为 6 个桶：`[128, 256, 512, 1024, 2048, 4096]`）。

---

## 4. 架构师决策指南与可证伪问题

1. **架构师设计练习**：在 148-SM B200（74 个 2-CTA Pair）上运行 `M=4096, N=3072` 的 `256×256` 分块 GEMM（192 个 Tile），前两波次消耗 $2 \times 74 = 148$ 个 Tile，剩余 44 个 Tile（$11264$ 列）。请设计整数拆分方程 $a \cdot 192 + b \cdot 128 = 11264$ 且 $a + b = 74$，求出 $(a, b)$。
   - *解答*：$64a + 128(a+b) = 11264 \implies 64a + 9472 = 11264 \implies 64a = 1792 \implies a = 28, b = 46$。
2. **后续可证伪问题**：当 `b12x` MoE 的 `routed_rows` 紧邻 2 的幂次上方（如 `R = 2049` 桶化至 `4096`）时，额外分配的未路由清零/填充开销是否会在延迟敏感的小 Batch 下超过缓存命中收益？
