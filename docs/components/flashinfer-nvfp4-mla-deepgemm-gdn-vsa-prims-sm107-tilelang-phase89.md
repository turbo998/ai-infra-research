# FlashInfer NVFP4 MLA 短单元 Decode 与通用 Split-KV 归约、批量 DeepGEMM FP8、GDN 融合训练、SM90 VSA 重调优、PrimTS Keeps Softmax 与 2-CTA MLA 页表窗口预取、SM107 Split-K=8 BF16 GEMM、Kimi K3 延迟释放及 TileLang Bulk TMA 缓存提示 / NVRTC TVM-FFI 深度解析（Phase 89）

> **English Version**: [flashinfer-nvfp4-mla-deepgemm-gdn-vsa-prims-sm107-tilelang-phase89.en.md](flashinfer-nvfp4-mla-deepgemm-gdn-vsa-prims-sm107-tilelang-phase89.en.md)  
> **证据等级**: `implementation`（固定 commit 源码审阅）+ `author-benchmark`（上游 PR 实测数据，非本项目硬件重测）+ `own-offline`（`experiments/flashinfer_nvfp4_mla_deepgemm_gdn_vsa_prims_sm107_tilelang_phase89/run.py` 纯标准库建模）

---

## 1. 背景与架构师问题定义

在 NVIDIA Blackwell（SM100 B200、SM103 B300/GB300、SM107）与 Hopper（SM90 H200）算子栈中，当 Tensor Core 的 MMA 峰值吞吐继续翻倍时，真实推理/训练延迟瓶颈往往转移到 **TMEM/SMEM 生命周期串行依赖、描述符页表全局内存（GMEM）回查延迟、小 Batch/窄矩阵 SM 占用率不足、以及 JIT 编译缓存键污染**。本篇基于 2026-10-26 合入 `flashinfer-ai/flashinfer` 与 `tile-ai/tilelang` 的 10 个非重叠一手 commit，系统拆解其实现机制、数学边界与云架构选型影响。

---

## 2. 核心算子与编译器路径逐项剖析

### 2.1 `cake_nvfp4_mla_decode`：短单元 Decode 与通用 Split-KV 归约（`flashinfer@5e8921c0d5`, PR #7418）
- **实现路径**：`flashinfer/cake_nvfp4_mla_decode/csrc/cake_nvfp4_mla_decode.cu` 针对 NVFP4 块缩放（Block-Scaled）MLA Decode 引入两项关键优化：
  1. **短单元（Short-Unit）分支**：当单请求 KV 长度满足 $\text{KV}_{\text{len}} \le 4096$（SM100 B200）或 $\text{KV}_{\text{len}} \le 8192$（SM103 GB300）时，收紧主循环流水线收尾 barrier 与累加器清零开销；
  2. **通用多路 Split-KV 归约（`split_kv >= 3`）**：取代原本仅针对 2 路特化的慢速回退路径，支持每行 $\ge 3$ 个 KV 分片的向量化 LSE（Log-Sum-Exp）重缩放与输出合并。
- **作者实测（`author-benchmark`）**：
  - **GB300（SM103）**：3-way split 从 `12.53 us` 降至 `11.30 us`（**`1.109x`**）；2-way split 从 `11.94 us` 降至 `11.20 us`（**`1.066x`**）。
  - **B200（SM100）**：3-way split 从 `13.04 us` 降至 `11.68 us`（**`1.116x`**）；2-way split 从 `12.32 us` 降至 `11.71 us`（**`1.052x`**）。

### 2.2 `cake_batch_deepgemm_fp8`：SM100/SM103 掩码批量 FP8 GEMM（`flashinfer@e23951f199`, PR #7425）
- **实现路径**：针对 MoE Decode 常见的 `masked_BatchGemm`（每个 Expert 有效行数由动态掩码 `masked_m` 决定），重构 CTA 持久调度（Persistent Tile Scheduler）与 TMA 块缩放因子预取，在跳过空 Expert tile 时消除无效 barrier 等待。
- **作者实测（`author-benchmark`）**：覆盖 164 个生产形状，在 B200（51 个形状更快）、GB300 #1（54 个更快）、GB300 #2（51 个更快）上实现 **0 退化**：
  - `b128_m128_n512_k128`：GB300 #2 从 `49.41 us` 降至 `13.86 us`（`PR/main = 0.2804`，**`3.565x` 加速**）；B200 从 `34.62 us` 降至 `14.11 us`（**`2.454x` 加速**）。
  - `sglang_p10_b32_m256_em1_n4096_k7168_ep8`：B200 从 `5.18 us` 降至 `2.46 us`（**`2.106x`**）；GB300 从 `4.99 us` 降至 `2.40 us`（**`2.079x`**）。

### 2.3 `cake_gdn`：Gated DeltaNet 训练前向准备、Tape 存储与融合 `dq/dk/dg`（`flashinfer@445980cc61`, PR #7423）
- **实现路径**：
  - 在 `flashinfer/experimental/cake_gdn_train/` 中将前向准备、状态磁带（State Tape）写出及反向 `dq/dk/dg` 融合进统一持久内核；动态共享内存（`dynamic_smem_bytes`）从 `91,136 B` 调整至 `99,328 B`。
  - **64 位整数溢出防护**：引入 `CheckedMul` 检查 `(total_tokens, H, D, K)` 步幅乘积，防止超长上下文 CP（Context Parallelism）训练时发生 32 位索引回绕。
- **作者实测（`author-benchmark`）**：在 `(H_qk, HV) = (16, 48), D = 128, CP ∈ {2, 4, 8, 16}`（Contiguous / Zigzag / Per-Doc Ragged）共 60 组配置上，GB300 前向几何平均 **`1.064x`**、完整训练步 **`1.042x`**；GB200 前向 **`1.067x`**（最高 `1.10x`）、训练步 **`1.043x`**。

### 2.4 `cake_vsa_sm90`：Hopper FP8 GQA 紧凑/共享转置器队列调优（`flashinfer@32a26fe723`, PR #7412）
- **实现路径**：针对 SM90 Video Sparse Attention（VSA）FP8 GQA 算子，发现过激的 L2 TMA 预取与 V 转置器 `ldmatrix` 并发块数反而挤占 SM90 寄存器与 L2 带宽：
  - 紧凑静态程序（Compact Static）：L2 预取距离从 `2 -> 1`，V 转置器在途 `ldmatrix` 块数从 `4 -> 1`（`g=8` 延迟从 `51.824 us` 降至 `49.824 us`，**`1.0402x`**）；
  - 共享转置器队列程序（Shared-Transposer Queue）：关闭 L2 预取（`1 -> 0`）。

### 2.5 `prims-ts`：打断 Keeps Softmax 串行链与多 Token 分页 Decode 规划器（`flashinfer@7ccb13b8e7`, PR #6937）
- **核心痛点**：在双实例 `Q128/KV128` Keeps Tile 中，原本 $P$ 覆写 TMEM 中的 $S$，导致下一轮 $QK^T$ 必须等待当前 $PV$ 消费完 $P$，Tensor Core 在 Softmax 期间完全空转。
- **四项解耦机制**：
  1. **16-bit K/V 分块移交**：Softmax 将 $P$ 按 4 个 `K32` 片段交给 $PV$ MMA，首个片段就绪即可启动 $PV$，并在 SM100 上将部分指数运算分流至 FMA 流水线；
  2. **8-bit K/V（FP8 / Sage）寄存器驻留**：$P$ 改写 SMEM 而非 TMEM，Softmax 在寄存器内保留分数，Max Pass 结束后立即释放 $S$，使下一轮 $QK^T$ 与当前指数计算并行；
  3. **锚点阈值跳重缩放（`softmax_rescale_threshold_log2`）**：当行最大值增长不超过阈值 binades（默认 8）时保持指数锚点不变，跳过绝大多数对输出累加器 $O$ 的就地重缩放；
  4. **分页多 Token Decode 规划器**：将 Dense/BF16 多 Token 分页请求从逐 Token 重读 KV 的 `Swaps Q8/Q32` 切换为分组 `Keeps Q128`。
- **作者实测（`author-benchmark`）**：
  - Dense `Q128/KV128`（`B=2, S=75600, H=40, D=128`）：BF16 在 B200 上 **`1.29x`**（`336.6 -> 261.9 Mcycles`），在 GB300 上 **`1.42x`**（`293.6 -> 206.5 Mcycles`）；Sage `k1` 在 GB300 上 **`1.30x`**。
  - Paged Decode 规划器切换（`dense BF16, Hq/Hkv=8, Sq=256, B=64`）：B200 从 `26,372 us` 降至 `4,786 us`（**`5.5x`**）；GB300 从 `25,797 us` 降至 `4,626 us`（**`5.6x`**）。

### 2.6 `prims-ts`：FP8 2-CTA Dense MLA 页表窗口寄存器预取（`flashinfer@b6694615b3`, PR #7019）
- **超线性退化根因**：在 `throughput_2cta` FP8 Dense MLA Decode 中，K/V 加载 Warp 原本在获取空闲 SMEM 槽位**之后**才去全局内存（GMEM）读取页表（Block Table），再发起 TMA。NCU 分析显示在 `B64, K8192, page=32` 下，QK MMA Warp 有 **49%** 时间停顿在等待 K，而 K 加载器有 **19%** 时间停顿在页表 GMEM 读取。
- **寄存器双窗口预取**：每个 K/V 加载 Warp 维护每 Lane 1 项（每 Warp 32 项）的页表寄存器窗口（`cached_window_page` 与 `cached_next_window_page`），覆盖 $32 / \text{pages\_per\_k\_tile}$ 个逻辑 K Tile（要求 $\text{pages\_per\_k\_tile} \mid 32$）。在当前窗口启动时即预取下一窗口，并在等待 SMEM 槽位前读取页 ID。
- **作者实测（`author-benchmark`，B200 `H128, K8192`）**：
  - `page=16, B=128`：从 `339.73 us` 降至 `141.71 us`（**`2.397x` 加速**）；
  - `page=32, B=128`：从 `198.31 us` 降至 `135.78 us`（**`1.461x` 加速**）。

### 2.7 SM107 BF16 Cluster Split-K=8 GEMM、Kimi K3 延迟释放与 TileLang TMA/NVRTC（PR #7064, #7411, #3683, #3285）
- **SM107 Split-K=8 与 18 级流水（`flashinfer@39c34abafc`, PR #7064）**：利用 SM107 更大的每 CTA 共享内存容量，将 `_MAX_AB_STAGES` 从 `12` 提升至 `18`，并在 `M <= 32` 的窄长 K 矩阵（如 `muse-spark pooler_slice TP4 1024×8192`）启用 `split_k = 8` 跨 CTA 簇归约，实现 `5%–8%` 延迟下降（`M=8` 从 `4.94 -> 4.61 us`）。同时实测证明非可移植的 `split_k ∈ {12, 16}` 反而慢 `1.3x–11x`，予以排除。
- **Kimi K3 AttnRes 延迟释放（`flashinfer@0aec9936bd`, PR #7411）**：在 SM103（B300/GB300）上为 `K=5` Dense 与 `K=1` Snapshot-Write 路由引入 Held-Release（延迟到 Chunk 末尾再释放已消费缓冲区），避免过早释放导致的流水线抖动，GB300 `M=1024, K=5` 达 **`1.0236x`**。
- **TileLang Bulk TMA 缓存策略与 NVRTC TVM-FFI（`tilelang@8ab967ff46` #3683, `tilelang@55743ac373` #3285）**：
  - 修复无描述符 Bulk TMA 加载（`cp.async.bulk.shared::cta.global`）的 `CacheHintSm90`（`EVICT_NORMAL` / `EVICT_FIRST` / `EVICT_LAST`）模板传递，保障 DeepSelect 流式路径正确控制 L2 驻留；
  - 支持基于 TVM-FFI 的 NVRTC 设备端直接编译，并将 `(compiler, compiler_version, options, target_code)` 納入 `cuda_binary_cache.py` 哈希键，杜绝不同 CUDA/NVRTC 版本或 `--use_fast_math` 编译产物的静默混用。

---

## 3. 创新架构横向对比（Fireworks / Together / Cerebras / Groq）

| 维度 | NVIDIA Blackwell (B200/GB300/SM107) + FlashInfer/TileLang | Fireworks AI / Together AI（云推理加速栈） | Cerebras WSE-3（晶圆级数据流） | Groq LPU（确定性 VLIW SRAM） |
| :--- | :--- | :--- | :--- | :--- |
| **多步/分页注意力瓶颈** | 通过 PrimTS `Keeps Q128` 与 32 项页表寄存器预取消除 TMA 页表停顿（最高 `5.6x` / `2.40x`） | 强依赖分组 Q Tile 与投机解码 Tree Attention 降低 KV 带宽放大 | 片上统一 SRAM 无 HBM 页表间接寻址开销，但受限于片上总容量 | 编译器静态排布 SRAM 流，无动态页表查找，但长上下文需多卡切分 |
| **稀疏/批量 MoE GEMM** | `cake_batch_deepgemm_fp8` 动态掩码调度实现 `3.56x` 加速，SM107 `split_k=8` 补齐小 Batch 利用率 | 在多租户 LoRA/MoE 路由中依赖此类掩码 Grouped GEMM 稳住 P99 TPOT | 稀疏激活可通过数据流网格路由，但动态 Expert 负载不均仍需重平衡 | 静态确定性流水难以跳过运行时空的 Expert 分支 |

---

## 4. 架构师练习与可证伪问题

1. **计算练习（2-CTA MLA 页表窗口周期）**：给定逻辑 K Tile 大小 $T_K = 128$ tokens，Warp 寄存器窗口容量 $W = 32$ 个页表项。当 $\text{page\_size} = 16$ 与 $\text{page\_size} = 64$ 时，$\text{pages\_per\_k\_tile}$ 分别为多少？每个 32 项窗口可覆盖多少个 K Tile？为什么 $\text{page\_size} = 24$ 会被 `make_mla_decode_config` 拒绝？
   - *参考推导*：$\text{pages\_per\_k\_tile} = 128 / 16 = 8$（每窗口覆盖 $32 / 8 = 4$ 个 K Tile）；$\text{page\_size}=64$ 时为 $2$（覆盖 $16$ 个 K Tile）。若 $\text{page\_size}=24$，则 $128$ 不能被 $24$ 整除且 $\text{pages\_per\_k\_tile}$ 无法整除 $32$，违反整 Tile 窗口对齐不变量。
2. **可证伪后续问题**：在 PrimTS `Keeps Q128` 16-bit K/V 路径中，当前代价模型（Cost Model）的 `mainloop factor = 3.2` 是在合入 K32 分段移交优化**之前**标定的；若重新按优化后延迟标定该系数，在 `Sq ∈ [4, 16]` 的中小长度多 Token Decode 上是否还能进一步扩大 `Keeps Q128` 相比 `Swaps` 的胜出区间？
