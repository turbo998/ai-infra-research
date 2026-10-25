# FlashInfer SM100/SM103/SM107 MXFP8 门控激活、NVFP4 量化+GEMM、Kimi-K3 路由与 AttnRes、SM107 QSA、AG+GEMM 与 Norm-Combine 深度解析（Phase 85）

> **关联实验**：`experiments/flashinfer_sm10x_mxfp8_nvfp4_kimi_qsa_ag_norm_phase85/`  
> **关联问题**：`q-flashinfer-sm10x-mxfp8-nvfp4-kimi-qsa-ag-norm-phase85` (`priority: 94`)  
> **English Version**: [`flashinfer-sm10x-mxfp8-nvfp4-kimi-qsa-ag-norm-phase85.en.md`](./flashinfer-sm10x-mxfp8-nvfp4-kimi-qsa-ag-norm-phase85.en.md)

---

## 1. 背景与架构演进概览

在 NVIDIA Blackwell（`sm_100a` B200，148 SMs）、Blackwell Ultra（`sm_103a` B300 148 SMs / GB300 152 SMs）以及 Rubin 早期架构目标（`sm_107a` R200，208–212 SMs）的算子协同优化中，FlashInfer 近期合并了 7 项核心 PR，系统性解决了极小 Batch Wave 利用率不足、超大 Shape 下每线程寄存器摊销不足、多节点 All-Gather + GEMM 尾部 Tile 闲置，以及 `cc=10.7` 15-bit SMEM 描述符兼容问题：

1. **`cake_gated_act_mxfp8` 窄分块与大 Shape 4 行/线程程序（PR #7346，`0b352a63`）**：覆盖 `sm_100a`、`sm_103a` 与 `sm_107a` 的 `silu_and_mul_mxfp8_quantize` 前向与反向（6 种模式：`fwd/bwd × rowwise/colwise/both`）。
2. **`cake_all_gather_matmul` SM103a 末轮 N-Split Twin（PR #7360，`e7f5b1e1`）**：针对 `world_size=8`、`M=1025`（对齐至 `1152`）、`N=1280, K=8192` 的 Llama-3.1-70B TP8 QKV 投影。
3. **`cake_mm_fp4` 4-CTA 行切分 Per-Token NVFP4 量化器与 `N16_SPLITK3_M` GEMM（PR #7356，`941fda25`）**：针对 `sm_100a` / `sm_103a` 长行（`K=7168..28672`）与窄 Token 场景。
4. **`cake_kimi_k3_fused_router` 全 28 格重生成调度（PR #7355，`ee50fd01`）**：针对 Kimi-K3（896 专家，`top-16`）在 `sm_100a` 与 `sm_103a` 上的全部 28 个 `(num_tokens, block_m)` 网格。
5. **`cake_qsa_nvfp4` SM107 (`cc=10.7`) 3-Wave 持久遍历与 15-bit SMEM 描述符（PR #7353，`1cfd7312`）**：将 `decode_tcgen05_persistent` 扩展至 R200（`sm_107a`）。
6. **`cake_warp_decode` 5 深工作环 K256 FC2 与 4 特征/线程向量化 Finalize（PR #7357，`ea95347a`）**：针对 `H=4096, E=512, top-10` 路由 MoE Decode。
7. **`cake_fused_norm_combine` 6 程序分发与 `cake_kimi_k3_attn_res` 双 CTA/SM 及无栅栏 K2/K3 调度（PR #7352 `20ede145` & PR #7338 `e9261a5e`）**。

---

## 2. 核心实现路径与数学/硬件公式

### 2.1 `cake_gated_act_mxfp8`：窄分块（`32x32` / `32x64`）与 4 行/线程程序 (`0b352a63`)
算子输入为 BF16 `[M, 2K]`（反向额外输入 BF16 `dOut [M, K]`），输出为 MXFP8 E4M3 数据与 `128x4` swizzled E8M0 块缩放因子 `[M, K]`：
- **小问题（Small $M \times K$）**：在第一阈值以下，使用窄 **`32x32`** CTA Tile（`sm_100a`/`sm_103a` 的 `fwd` 以及全部 3 个架构的 `bwd`），避免 `128x128` Tile 在小矩阵上只启动极少数 CTA 导致 SM 闲置。
- **GB300 (`sm_103a`) 中等区间**：`colwise fwd` 在 $2^{22} = 4,194,304$ 到 $2^{24} = 16,777,216$ 元素区间使用 **`32x64`** Tile。
- **大问题（Large $M \times K$）**：在 `sm_107a` 大尺寸及 `sm_103a` 最大 `fwd both` 尺寸（如 `131072x8192`）上，切换至 **每线程 4 行（four rows per thread）** 程序，摊销指令发射与地址计算开销。
- **实测加速（Cold L2）**：
  - B200 (`sm_100a`, 148 SMs)：`128x128` 6 种模式提速 **`1.144x–1.260x`**（`fwd colwise`: `2.67 -> 2.12 us`，**`1.260x`**）。
  - GB300 (`sm_103a`, 152 SMs)：`128x128 fwd colwise` `2.26 -> 2.02 us`（**`1.121x`**），`131072x8192 fwd both` `923.27 -> 904.89 us`（**`1.020x`**）。
  - R200 (`sm_107a`, 208 SMs)：`128x128 bwd colwise` `2.42 -> 2.17 us`（**`1.116x`**），`131072x8192 fwd rowwise` `528.30 -> 481.99 us`（**`1.096x`**）。

### 2.2 `cake_all_gather_matmul`：SM103a 末轮 Peer N-Split Twin (`e7f5b1e1`)
在 `world_size = 8`、`M = 1025` rows/rank（`M_padded = 1152`，每 rank $1152 / 128 = 9$ 个 M-tile）、`N = 1280`（$1280 / 256 = 5$ 个 256 宽 N-tile）、`K = 8192` 的 K-major BF16 All-Gather + GEMM 中：
- 前 7 个 Peer Pass 正常流水执行；到了最后一个 Peer Pass，剩余 $P = 9 \times 5 = 45$ 个 `128x256` Tile。
- 当 $2P \le \text{num\_ctas}$（$90 \le 148$）时，新内核 `main_bfloat16_ws8_k_major_cg1s2`（`192` 线程，`197,632 B` 动态 SMEM，`TMEM_NCOLS = 512`）将末轮 $P = 45$ 个 Tile 沿 $N$ 维对半切分为 **$2P = 90$ 个 `128x128` 半宽 Tile**，把原本闲置的 45 个 CTA 投入末轮计算，消除了尾部流水排空瓶颈。
- 8x B300 实测：Node 1 从 `341.6 -> 336.6 us`（**`1.0116x`**），Node 2 从 `340.7 -> 337.7 us`（**`1.0209x`**）。

### 2.3 `cake_mm_fp4`：4-CTA 行切分量化器与 `N16_SPLITK3_M` 驻留门控 (`941fda25`)
- **4-CTA Cluster 行切分（`_cs4`）**：当 `M` 很小（如 `M=1..17`）但 `K` 很大（如 `K=28672`）时，单行量化由 4 个 CTA 组成的 Cluster 协同切分完成：B200 量化器最高提速 **`1.255x`**（`K=28672, M=17`: `3.601 -> 2.874 us`），GB300 最高提速 **`1.243x`**（`K=28672, M=1`: `3.235 -> 2.604 us`）。
- **`N16_SPLITK3_M` 门控**：在 `K=7168` 窄行（如 `M=64, N=2048`）下，原本 `8`-token Tile + `SPLITK3_CLUSTER=3` 会产生 $(64/8) \times (2048/256) \times 3 = 192$ 个 CTA，超过 B200 (`148` SMs) / GB300 (`152` SMs) 单波驻留上限；切换至 `16`-token Tile 后僅需 $(64/16) \times 8 \times 3 = 96$ 个 CTA（单波驻留），同时在大矩阵 `K=8192, N=28672, M=8192` 上 GB300 GEMM 达 **`1.144x`**（`519.5 -> 454.5 us`），Fused `quant+GEMM` 达 **`1.140x`**（`540.6 -> 474.0 us`）。

### 2.4 `cake_kimi_k3_fused_router` 与 `cake_qsa_nvfp4` SM107 持久遍历 (`ee50fd01`, `1cfd7312`)
- **Kimi-K3 Fused Router (`ee50fd01`)**：针对 `896` 专家、`top-16` 路由的全 28 个 `(num_tokens, block_m)` 单元全部重生成：GB300 (`sm_103a`) 提速 **`1.10x–1.80x`**（`tokens=16, block_m=8`: `5.89 -> 3.39 us` **`1.74x`**；`tokens=8192, block_m=16`: `40.66 -> 22.61 us` **`1.80x`**）；B200 (`sm_100a`) 提速 **`1.06x–1.73x`**。
- **SM107 QSA Persistent Walk (`1cfd7312`)**：
  1. 在 `__CUDA_ARCH__ == 1070` 下，UMMA SMEM 描述符基址掩码从 14-bit (`0x3FFF`) 扩展为 **15-bit**：`(((smem_q_addr) >> 4) & 0x7FFF)`。
  2. 引入 **3-Wave 上界公式**：
     $$1 < \left\lceil \frac{\text{rows} \times \text{num\_kv\_heads} \times \text{plain\_splits}}{\text{sm\_count}} \right\rceil \le 3$$
     在 R200（`212` SMs）、`rows=128, num_kv_heads=2, plain_splits=2`（512 work items）下，$\lceil 512 / 212 \rceil = 3 \le 3$，每个 `(row, KV head)` 仅分配 `2` 个 partial 槽位即可持久遍历合并，R200 HND 延迟从 `55.71 -> 46.05 us`（**`1.21x`**），NHD 从 `54.99 -> 47.15 us`（**`1.17x`**）。

### 2.5 `cake_warp_decode`、`cake_fused_norm_combine` 与 `cake_kimi_k3_attn_res` (`ea95347a`, `20ede145`, `e9261a5e`)
- **`cake_warp_decode` (`ea95347a`)**：`LinearK256Fc2` 将 `NUM_WORK_PIPE_STAGES` 从 `3` 增至 **`5`**，并在 SMEM 中增加 `512` 行路由元数据缓存表（`8,192 B`，总 SMEM 从 `179,456 B -> 187,648 B`）；Top-10 Finalize 改为每线程处理 `4` 个连续特征（Grid 缩小 `4x`），在 `h4096_e512_i256_k10` (`M=4`) 上 B200 提速 **`1.113x`**（`23.58 -> 21.19 us`），GB300 提速 **`1.104x`**（`22.87 -> 20.72 us`）。
- **`cake_fused_norm_combine` (`20ede145`)**：8-rank 集合归约 + 双路 RMSNorm（`H=2560`）由 4 程序扩展为 **6 程序**（`tokens=1` 独立构建，`593..1772` tokens 采用 `3 CTAs/SM` 流水程序），`tokens=1` 在 B200 提速 **`1.088x`**（`8.736 -> 8.032 us`）、B300 提速 **`1.092x`**（`9.153 -> 8.384 us`）。
- **`cake_kimi_k3_attn_res` (`e9261a5e`)**：针对 `H=7168, K \in [0, 8]` 的残差候选打分+Softmax+RMSNorm 融合算子，新增 **2-CTA/SM (`_cta2`)** 与 **无 consumed-release 栅栏 (`_crfnone`)** 持久程序，程序库从 `138`（167 镜像）扩展至 **`181`（211 镜像）**，最高提速达 **`1.1297x`**（B200 `M=192, K=2`）、**`1.1219x`**（B300 `M=192, K=3`）与 **`1.1097x`**（GB300 `M=192, K=3`）。

---

## 3. 架构师设计练习与可证伪问题

1. **计算练习（All-Gather + GEMM 末轮 N-Split 门控）**：当 `M_padded = 1152, N = 1280` 时，基础 Tile 为 `128x256`，末轮共有 $P = (1152/128) \times (1280/256) = 45$ 个 Tile。若 `N` 扩大至 `2560`（$P = 90$），在 148 SM 的 B300 上 $2P = 180 > 148$，为什么此时不应开启 N-Split？（提示：超出驻留 CTA 数会引入第二波尾效应并翻倍权重重复加载）。
2. **可证伪假设**：在 `__CUDA_ARCH__ == 1070` 上，若继续使用 14-bit SMEM 描述符掩码 `0x3FFF`，当 `smem_q_addr >= 262,144 B`（`0x4000 << 4`）时，高位地址位会被截断并导致 UMMA 读取错误的共享内存偏移。
