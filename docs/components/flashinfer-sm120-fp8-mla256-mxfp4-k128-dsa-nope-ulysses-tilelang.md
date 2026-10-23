# FlashInfer SM120 FP8 256/256 MLA & GQA Prefill、Hopper MXFP4×FP8 K128 Folded MegaMoE、Blackwell/Rubin 原生 NoPE 稀疏 MLA 训练、Cake Ulysses 交错条带 NVLink All-to-All 与 TileLang Top-K / 2:4 稀疏元数据双射护栏深度解析

> **关联研究问题**：`q-flashinfer-sm120-fp8-mla256-mxfp4-k128-dsa-nope-ulysses-tilelang`（Priority 69，Phase 60 Track A）  
> **覆盖上游源码提交**：
> - `flashinfer-ai/flashinfer@8e8342f909d4422c67d3e83b94aa8a5e94a68824` (`#6466`, 2026-10-23)
> - `flashinfer-ai/flashinfer@b17eaa011241866e45a3ea06797c5fecaf6c7bc7` (`#6465`, 2026-10-23)
> - `flashinfer-ai/flashinfer@592deecaff131b07bff5d53c4f61c8f95621f65a` (`#6929`, 2026-10-22)
> - `flashinfer-ai/flashinfer@49aa24fbc4b30a2ba0be61062342f891e8af2972` (`#6887`, 2026-10-20)
> - `flashinfer-ai/flashinfer@6a4e51eb81a108c0d44e32d74c7713da6577d92c` (`#6489/#6836`, 2026-10-20)
> - `flashinfer-ai/flashinfer@60b1652f903bc6169b7ac1a4c9968e42e58cd9b2` (`#6864`, 2026-10-20)
> - `tile-ai/tilelang@4da574c01371a79825c91492f09be6a83a4cb400` (`#3280`, 2026-10-22)
> - `tile-ai/tilelang@ced0ca1a30356d5834fe38b1d4257a09773e467a` (`#3125`, 2026-10-22)
> - `tile-ai/tilelang@1a52a1cd0628f59cbc45c5d29505fab0e9f08b92` (`#3595`, 2026-10-21)  
> **配套确定性实验**：`experiments/flashinfer-sm120-fp8-mla256-mxfp4-k128-dsa-nope-ulysses-tilelang/run.py`

---

## 1. 架构背景与核心问题

在 Blackwell 工作站/边缘 GPU（SM120/SM121）、Hopper（SM90a）以及 Blackwell/Rubin 数据中心 GPU（SM100a/SM103a/SM107a）的算子演进中，以下五个底层瓶颈直接决定了长上下文 Prefill、MoE 路由和序列并行训练的吞吐边界：

1. **SM120 FP8 Prefill Attention 的非对称 MLA 维度与 GQA 头映射缺失**：在 SM120（如 RTX PRO 6000 Blackwell / 5K Pro，110 SMs，每 SM 99 KiB 共享内存上限）上，此前 FMHA v2 Warp-Specialized FP8 E4M3-to-BF16 内核仅覆盖了 DeepSeek-V3.2 的非对称 `192/128` 维度与 MHA。面对 **GLM-4.7-Flash**（`QK/V = 256/256`，192 NoPE + 64 RoPE，20/20 heads）与 **GLM-4.7**（`QK/V = 128/128`，96 Q heads / 8 KV heads，$G=12$ GQA），缺乏对应的寄存器/SMEM 分块与 KV head 步长索引。
2. **Hopper MXFP4 × FP8 MegaMoE 在非 256 整除中间维上的调优退化**：当 MoE 模型的单专家中间维 $I = 640$（`gate_up = 1280`）时，由于 $640 \bmod 256 = 128 \neq 0$，所有要求 `K=256` 对齐的折叠（folded）候选被 `can_implement` 过滤，导致 11 个调优候选中仅剩 **3** 个可用（两个 `M128/K128` 和一个 `M256/N32/K128`），在大 batch 场景下遭遇严重的 wave quantization 与尾块浪费。
3. **无位置编码（NoPE）稀疏 MLA 训练的冗余 RoPE 计算与 HBM 读写**：**GLM-5.3-Flash** 的 DeepSeek Sparse Attention（DSA）省去了 64 维拼接 RoPE 子块，仅保留 512 维共享潜变量与 64 个查询头。若复用带 RoPE 的 576 维内核并通过零填充规避，不仅浪费 $64/576 = 11.11\%$ 的 HBM 带宽，还会执行无意义的 RoPE 反向归约。
4. **Blackwell NVLink 序列并行 All-to-All 的 L2 与跨卡写入争用**：Ulysses 序列并行的 `scatter_heads` / `gather_heads` 在单机 NVLink（2/4/6/8 卡）上若采用顺序分块写入，会导致多卡并发 DMA 命中远端同一 HBM 通道。
5. **TileLang Lightning Indexer Top-K 固定共享内存溢出与 sm8x 2:4 稀疏元数据非双射静默损坏**：DeepSeek-V3.2 Top-K 选择器将基数桶候选缓冲硬编码为 `SMEM_INPUT_SIZE = 4096`，当 `seq_len = 32768` 且大量分数相等时发生越界写；同时 `make_metadata_layout`（SM8x 2:4 稀疏）在 $M \bmod \text{group} \neq 0$ 或 $K \bmod 2 \neq 0$ 时产生非双射索引映射，被 Layout 推断器静默放行。

---

## 2. 核心实现机制与源码级剖析

```
+--------------------------------------------------------------------------------------------------+
| Phase 60 Track A: GPU Kernel & Quantization Pipeline Architecture                                |
+--------------------------------------------------------------------------------------------------+
| 1. SM120 FP8 E4M3->BF16 FMHA v2 (#6466)                                                          |
|    - GLM-4.7-Flash (256/256, 20/20 MHA): 233 regs/thread, 64 KiB SMEM, 1 CTA/SM -> 1.53x-1.61x   |
|    - GLM-4.7 (128/128, 96/8 GQA): head_idx = (qkv_offset==0) ? bidh : (bidh / h_q_per_kv)        |
|                                                                                                  |
| 2. Hopper MXFP4 x FP8 MegaMoE Folded K128 Candidates (#6465)                                     |
|    - I=640 (640 % 256 != 0) filters out K256; adds M256/N64/K128 (clusters 1x1x1 & 1x2x1)        |
|      + M256/N16/K128 -> eligible candidates 3 -> 8; -17.72% compute / -14.93% E2E at 1024 tok    |
|                                                                                                  |
| 3. Blackwell/Rubin Native NoPE DSA Training (#6836) & Cake Ulysses A2A (#6864)                   |
|    - cake_dsa_h64_nope_train (512 latent, dkv = dK + dV in FP32 [S,512]): 1.74x fwd on GB300     |
|    - Interleaved-stripe NVLink scatter/gather (24 variants): 1.33x-2.50x on B200/B300            |
|                                                                                                  |
| 4. TileLang Guardrails (#3280, #3125, #3595)                                                     |
|    - Dynamic global workspace [B, 2, seq_len] replaces fixed 4096 SMEM buffer for >4K ties       |
|    - sm8x 2:4 sparse metadata enforces M % group == 0 (32/16) & K % 2 == 0 for bijection         |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 SM120 FP8 256/256 MLA 与 GQA Prefill Attention (`flashinfer@8e8342f909`)

在 `csrc/fmha_v2/fmha/smem_tile_o.h` 与 `fused_multihead_attention_kernel_1xN_reload.h` 中，FlashInfer 为 SM120 引入了两条关键扩展：

1. **GLM-4.7-Flash `256/256` Warp-Specialized 流水线**：
   - QK 与 V 头维度均为 `256`（对应 `4` 个 `64` 维 MMA K-step）。为适配 SM120 每 SM 99 KiB 的共享内存硬限制，内核将动态 SMEM 严格控制在 **64 KiB**（1 CTA/SM），FP8 路径每线程寄存器占用为 **233 regs**（作为对比，同几何结构的 BF16 reload 路径占用 **247 regs** 且 spilled 64B stack frame）。
   - 在 `Smem_tile_o_row_major` 中，针对 `BYTES_PER_STG == 16`（8 个 BF16 元素/线程），将累加器对映射到连续 16B 行，使 `HEAD_DIM_V = 256`（32 组 quad pairs）能够直接通过 `16B STG.128` 落盘，消除共享内存 bank 冲突。
2. **GLM-4.7 GQA 独立 KV 头步长索引**：
   - 在 `Fused_multihead_attention_params_v2` 中新增 `h_kv` 与 `h_q_per_kv = h / h_kv`。
   - 在 `Hopper_qkv_tile_params` 构造函数中，根据 `qkv_offset` 区分 Q 与 K/V 的 head 索引：
     $$\text{head\_idx} = \begin{cases} \text{bidh}, & \text{if } \text{qkv\_offset} = 0 \text{ (Query)} \\ \lfloor \text{bidh} / h_{\text{q\_per\_kv}} \rfloor, & \text{if } \text{qkv\_offset} \in \{1, 2\} \text{ (Key / Value)} \end{cases}$$

### 2.2 Hopper MXFP4 × FP8 MegaMoE K128 折叠调优候选 (`flashinfer@b17eaa0112`)

在 `flashinfer/moe_ep/hopper_mxfp4_megamoe.py` 中，此前折叠（folded）瓦片 `M256/N64` 仅绑定了 `K256`（因为将 `K128` 减半至 `K64` 会破坏折叠主循环不变量）。但实际上 `M256/N64/K128`（折叠后为 `M128/N128/K128`，4 pipeline stages）完全满足 Hopper WGMMA 约束且仅需 $I \bmod 128 = 0$。该提交新增了 5 个 `K128` 候选：
- 3 个 `M256/N64/K128` 单 CTA 簇 `(1,1,1)` 候选（`group_hint` 分别为 `132`、`264`、`330`，对应 H200 132 SM 的 1.0x/2.0x/2.5x 波次）；
- 1 个 `M256/N64/K128` 双 CTA 簇 `(1,2,1)` 尾块配对（tail-pair）候选；
- 1 个 `M256/N16/K128` 细粒度 N 瓦片候选（适配中低 token 密度）。

对于 $I = 640$ 的模型，合法调优候选从 **3 个跃升至 8 个**。

### 2.3 Blackwell/Rubin 原生 NoPE 稀疏 MLA 训练 (`flashinfer@6a4e51eb81`) 与 Ulysses 交错条带 A2A (`flashinfer@60b1652f90`)

- **原生 NoPE 稀疏 MLA 训练（`cake_dsa_h64_nope_train`）**：专为 GLM-5.3-Flash（64 头，512 维共享 KV 潜变量，0 维 RoPE）打造，覆盖 `sm_100a`（B200/GB200）、`sm_103a`（B300/GB300）和 `sm_107a`（Rubin R200）。由于 $K$ 与 $V$ 共享同一个 `[S, 512]` 潜变量张量，反向传播直接将梯度合并累加为单一 FP32 张量 $\text{dkv} = dK + dV \in \mathbb{R}^{S \times 512}$，彻底消除了 64 维 RoPE 分支的 `tma_red_f32` 原子归约与寄存器切分。
- **Cake Ulysses 交错条带 NVLink All-to-All（`#6864`）**：在 `sm_100a`/`sm_103a` 上实现了 24 个内核变体（3 种精度 `bf16/fp16/fp32` × 2 个方向 `scatter_heads/gather_heads` × 4 种通信域大小 `2/4/6/8 ranks`），采用交错条带（interleaved-stripe）对等指针表 `CakePeerPointerTable` 与信号同步，包含 staging-to-output 拷贝在内，在 8×B200 上取得 **1.33x–2.49x** 加速，在 B300 上取得 **1.29x–2.50x** 加速。
- **Cake BF16 Fused QK RMSNorm + NeoX RoPE + Paged KV Append（`#6887`）**：统一了 SM90/SM100/SM103 的单内核发射路径，在 28 个基准形状上取得 H100 **1.0623x**、H200 **1.0738x**、B200/B300/GB300 **1.14x** 的几何平均加速（`B=1, Q=1` 解码单步最高达 **1.2109x**）。

### 2.4 TileLang Top-K 越界修复、2:4 稀疏双射校验与寄存器级 BF16x2 转换 (`tilelang@4da574c013`, `ced0ca1a30`, `1a52a1cd06`)

1. **DeepSeek-V3.2 Top-K 选择器越界修复（`#3280`）**：在 `examples/deepseek_v32/top_k_selector.py` 中，基数排序对 8-bit 桶扫描找到阈值桶 ` threshold_bin` 后，将落入该桶但尚未确定排名的候选索引收集至缓冲区。旧实现使用固定大小的共享内存 `s_input_idx[2, 4096]`；当 `seq_len > 4096`（如 32768）且大量位置得分相同时，候选数远超 4096 导致共享内存写穿（write-out-of-bounds）。修复方案将双缓冲候选队列移至全局工作区 `workspace[batch, 2, seq_len]`，容量严格与 `seq_len` 绑定。
2. **SM8x 2:4 稀疏元数据布局双射校验（`#3125`）**：`make_metadata_layout` 构造的索引映射为：
   $$ii = \lfloor i / G \rfloor \cdot G + (i \bmod 8) \cdot W + \lfloor (i \bmod G) / 8 \rfloor$$
   其中 16-bit 下 $(G, W) = (32, 4)$，8-bit 下 $(G, W) = (16, 2)$，随后在 $2 \times 2$ 块内做副对角线交换。当 $M \bmod G \neq 0$ 或 $K \bmod 2 \neq 0$ 时，该映射不再是双射（bijective），会导致 `FlattenedLayout::get_flattened_extent()` 与原形状 $M \times K$ 失配或越界。`ced0ca1a30` 在 `make_metadata_layout` 入口增设 `ICHECK_EQ(m % group, 0)` 与 `ICHECK_EQ(k % 2, 0)`，并在 `LayoutNode::DebugOutput` 中加入双射校验断言。
3. **无置换指令的 `bf16x2 -> f32x2` 转换（`#3595`）**：`tl::__tl_cvt_bf16x2_to_f32x2` 使用两条纯位运算指令替代 PTX `prmt.b32`：
   ```cpp
   uint32_t lo = in_bits << 16;          // shl.b32: 低16位 BF16 左移至 FP32 高16位
   uint32_t hi = in_bits & 0xffff0000u;  // and.b32: 高16位 BF16 原位保留为 FP32 高16位
   ```

---

## 3. 量化实验与关键指标对比

基于本仓库确定性验证脚本 `experiments/flashinfer-sm120-fp8-mla256-mxfp4-k128-dsa-nope-ulysses-tilelang/run.py` 核验的上游实测数据汇总如下：

### 表 1：SM120 (5K Pro GPU, 110 SMs) GLM-4.7-Flash `256/256` FP8 Prefill 延迟对比 (`#6466`)

| Heads ($H_q/H_{kv}$) | Seq Len ($S$) | FP8 FMHA v2 ($\mu\text{s}$) | Matched BF16 FMHA v2 ($\mu\text{s}$) | BF16 FA2 ($\mu\text{s}$) | 相对 Matched BF16 加速比 | 相对 BF16 FA2 加速比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `20 / 20` (1 Node) | 1024 | **68.88** | 105.33 | 103.54 | **1.53x** | **1.50x** |
| `20 / 20` (1 Node) | 4096 | **782.40** | 1241.56 | 1110.74 | **1.59x** | **1.42x** |
| `20 / 20` (1 Node) | 8192 | **2975.75** | 4690.40 | 4077.53 | **1.58x** | **1.37x** |
| `5 / 5` (TP=4) | 1024 | **30.35** | 47.07 | 43.59 | **1.55x** | **1.44x** |
| `5 / 5` (TP=4) | 4096 | **243.48** | 387.44 | 377.05 | **1.59x** | **1.55x** |
| `5 / 5` (TP=4) | 8192 | **803.80** | 1290.77 | 1133.70 | **1.61x** | **1.41x** |

### 表 2：2×H200 (`EP=2, E=512, topk=10, H=2560, I=640`) MXFP4×FP8 MegaMoE K128 候选加速 (`#6465`)

| Tokens / Rank | 原候选集 Compute ($\mu\text{s}$) | 新增 K128 候选后 Compute ($\mu\text{s}$) | Compute 延迟降低 | 原候选集 E2E ($\mu\text{s}$) | 新增 K128 候选后 E2E ($\mu\text{s}$) | E2E 延迟降低 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 339.216 | **316.272** | **-6.76%** | 501.264 | **475.856** | **-5.07%** |
| 1024 | 593.264 | **488.128** | **-17.72%** | 756.176 | **643.296** | **-14.93%** |
| 4096 | 1405.728 | **1315.200** | **-6.44%** | 1573.472 | **1475.216** | **-6.24%** |
| 8192 | 2670.336 | **2343.024** | **-12.26%** | 2891.616 | **2560.176** | **-11.46%** |
| 16384 | 5223.024 | **4442.880** | **-14.94%** | 5628.496 | **4873.776** | **-13.41%** |

### 表 3：GB300 (`sm_103a`) 与 R200 (`sm_107a`) 原生 NoPE 稀疏 MLA 训练 vs Zero-Padded RoPE (`#6836`)

| 硬件架构 | 评测负载 ($S \times S_{kv}, K$) | Forward NoPE (ms) | Forward RoPE (ms) | Fwd 加速比 | Step NoPE (ms) | Step RoPE (ms) | Step 加速比 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GB300 (`sm_103a`)** | `doc_4096` ($K=2176$) | **0.716** | 1.246 | **1.74x** | **3.707** | 5.150 | **1.39x** |
| **GB300 (`sm_103a`)** | `doc_16384` ($K=128$) | **0.655** | 1.201 | **1.83x** | **2.443** | 4.774 | **1.95x** |
| **GB300 (`sm_103a`)** | `short_1024_k128` ($K=128$) | **0.049** | 0.558 | **11.43x** | **0.323** | 0.878 | **2.72x** |
| **GB300 (`sm_103a`)** | `glm53_a` ($16263 \times 269564, K=2176$) | **2.836** | 3.801 | **1.34x** | **15.674** | 20.181 | **1.29x** |
| **R200 (`sm_107a`)** | `doc_16384` ($K=128$) | **0.387** | 0.602 | **1.56x** | **1.508** | 3.515 | **2.33x** |
| **R200 (`sm_107a`)** | `glm53_a` ($16263 \times 269564, K=2176$) | **1.706** | 2.369 | **1.39x** | **9.109** | 13.923 | **1.53x** |

---

## 4. 解决方案架构师设计指南与反证问题

1. **SM120 工作站/边缘推理的 99 KiB SMEM 边界**：在 RTX PRO 6000 Blackwell 等 SM120 卡上部署 GLM-4.7-Flash（256/256 头维）时，BF16 累加器 reload 路径会触及 247 regs/thread 并产生 64B 局部内存溢出；启用 FP8 E4M3-to-BF16 FMHA v2 不仅利用了 2× Tensor Core 吞吐，还将寄存器压回 233 regs/thread、动态 SMEM 固定在 64 KiB，稳定兑现 **1.53x–1.61x** 加速。
2. **MoE 中间维 $I$ 与 Folded Tile 的整除性审计**：在设计细粒度专家（如 $E=512, I=640$）时，必须验证 $I \bmod 256$ 是否为 0。若仅为 128 的倍数（而非 256 的倍数），必须确保底层算子库注册了 `M256/N64/K128` 折叠瓦片与 `(1,2,1)` 簇尾块配对，否则会在 `tokens_per_rank >= 1024` 时损失 15%–18% 的 GEMM 吞吐。
3. **后续可证伪问题**：在 Rubin `sm_107a` 上，原生 NoPE 稀疏 MLA 训练将 `dkv` 合并为单张量 `[S, 512]` FP32 原子归约后，当 $K/S \to 1$（稠密化极限）且多 CTA 并发写同一 KV 块时，L2 atomic `tma_red_f32` 是否会成为新的带宽瓶颈？
