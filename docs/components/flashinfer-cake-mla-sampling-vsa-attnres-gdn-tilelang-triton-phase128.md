# Phase 128 深度研究（Track A）：FlashInfer Cake NVFP4 MLA 解码预取、Sampling Round 17 整块尾/紧凑列表、Hopper VSA 位置锁定、Kimi-K3 TMEM 驻留累加器、GDN 异步准备与 TileLang/Triton 契约

## 1. 背景与核心问题

在 Blackwell（SM100/SM103/SM120）与 Hopper（SM90a）异构算子栈中，当核函数粒度收敛至微秒级时，**内核建立期首个工作项延迟**、**大 Top-K 双核启动开销**、**高寄存器压力下的局部内存溢出（Spill）** 以及 **上下文并行（CP）梯度传播的主机侧同步准备与前向磁带（Tape）驻留** 成为主要瓶颈。本报告基于 2026-10-28 最新合入的 11 个上游提交（`flashinfer-ai/flashinfer` 9 个、`tile-ai/tilelang` 1 个、`triton-lang/triton` 1 个），建立确定性算术与硬件资源模型。

## 2. 关键实现路径与机制剖析

### 2.1 Cake NVFP4 MLA 短单元解码建立期首项预取（`949a9f2d53ca`）
- **实现路径**：`flashinfer/experimental/nvfp4_mla_decode/{cake_backend.py,cake_jit.py}` 新增 `decode_short_spec_uh`（SM100）与 `decode_short_epi_warp_spec_uh`（SM100/SM103）。
- **核心机制**：当计划中的每个工作单元（Work Unit）仅包含单个项对（Item Pair）时，项 `2 * cluster` 必为该单元首项且位于元数据表内。新内核在 cluster 汇合屏障（Rendezvous）之前的内核 Setup 阶段即预加载 `2 * cluster` 的字段并立即启动 Load Warp，消除汇合后的首次元数据等待，在 B200 / GB200 短单元解码上分别取得最高 **`1.026x`** 与 **`1.029x`** 加速。

### 2.2 Cake Sampling Round 17 单次启动整 CTA 尾部与 SM12x 紧凑候选列表（`3812cddafdb7`）
- **实现路径**：`csrc/cake_sampling/cake_sampling_binding.cuh` 新增 `kFlagCompactList = 8192`，Stage-1 构建数从 `108 -> 135`（总核函数从 `123 -> 150`）。
- **核心机制**：
  1. 针对 `(8, 16)`、`(1, 16)`、`(2, 32)` 流式构建引入单次启动的 Leader-Push / Default-Sample 整 CTA 尾部（Whole-CTA Tail），在 `top_k > 768..800` 时消除 Stage-1 与 Stage-2/3 之间的二次核函数启动开销（B200 最高 **`1.125x`**、H100 **`1.121x`**、GB300 **`1.119x`**）。
  2. 针对 SM12x（RTX PRO 6000，188 SM，`101,376 B` 共享内存上限），新增 23 个每 CTA 8,192 候选槽位的紧凑列表构建（Compact-List Twin），使单波次流式采样在大词表（`128,256 / 151,936`）与 Batch `32..128` 下实现 **`1.307x–1.413x`** 提速。

### 2.3 Hopper Cake VSA 位置选择 Sage-FP8 GQA 静态核（`279f9a082fb2`）
- **实现路径**：`flashinfer/cake_vsa_sm90_backend/cake_planner.py` 新增 `PIN_MIN_CTA_LOOP_ITERS = 14` 与 `attention_fp8_gqa_pin`。
- **核心机制**：当最轻负载 CTA 的主循环迭代数 `cta_loop_iters >= 14` 时，切换至 `u16` 载体 `cvt.rn.satfinite.e4m3x2.f32` 打包与 V-ring 释放后立即执行 `mbarrier.test_wait` + `min(bias, fma(token, 1e30, bias))` 零值不变门控，防止汇编器将 V 释放指令下沉至 66 条 `ex2` 指令之后。主循环指令数从 `675 -> 633`（`-42` 条），20 个路由形状几何平均加速 **`1.0287x`**（最高 **`1.0361x`**），168 寄存器零 Spill 且输出逐位一致。

### 2.4 Cake Kimi-K3 AttnRes TMEM 驻留累加器与 FP8 投影双生核（`d0ac3f0b1649`, `21e02addf9f6`）
- **AttnRes TMEM 驻留累加器（`d0ac3f0b1649`）**：针对 `M=192, K=6..8` 的 384 线程对核，将两个 Consumer Warpgroup 的寄存器配额从 `232` 降至 `200`（Producer 从 `40` 增至 `104`），并在前缀块第一遍流式传输期间将每个 Consumer Stream 的 FP32 累加器驻留（Park）在额外的Tensor Memory 列中（每流 TMEM 从 `256 -> 512` 列），彻底消除局部内存寄存器溢出（Spill `>0 -> 0`），在 B200 与 GB300 上实现最高 **`+9.9%`** 提速（B300 **`+7.9%`**）。
- **Kimi-K3 FP8 投影（`21e02addf9f6`）**：新增由 MMA Warp 分配并在双 Epilogue 释放最后累加器后立即回收 TMEM 的双阶段 4 槽位 GEMM 双生核，以及 GB300 352 宽 `q_proj` TMA-Store 路由与 L2 `evict_first`/`evict_last` 量化提示，在 GB300 上分别取得 **`1.042x`**（`TP1 kv_b M=256`）、**`1.038x`**（`TP8 f_b M=4096`）与 **`1.027x`**（`TP1 q_proj M=512`）加速。

### 2.5 Cake GDN 训练反向预取、零同步准备与前向磁带自动复用（`b6089e61ae3f`, `025f5acf562f`）
- **反向 Adjoint Walk 预取与 LPT 分段（`b6089e61ae3f`）**：64 宽 `offset` 反向核提前一步预取两个 256-bit `dv/i` 载体并将块表放入双槽共享内存环（动态共享内存 `82,944 -> 83,968 B`），主机规划器默认启用最长处理时间（LPT）分段规则（GB300 CP4 zigzag **`1.04x`**，CP8 contiguous **`1.03x`**）。
- **零同步准备与磁带复用（`025f5acf562f`）**：将 `build_kcp_context` + `prepare` 中的 24 次可分页 `torch.tensor(..., device=cuda)` 同步拷贝合并为按 dtype 单一锁页（Pinned）主机缓冲区的非阻塞 `copy_`，并通过合并相同 SM 负载加速 LPT 二分水位线求解，使 `(2, 6)` 布局在队列繁忙下的准备耗时从 **`147.0 ms` 降至 `5.8 ms`（`25.34x` 加速，同步调用 `24 -> 0`）**；同时 `reuse_recompute_tapes="auto"` 仅当 GDN 为 Checkpoint 区域最后保存节点时移交前向磁带，否则走重放路径，将反向峰值显存从 **`9,251 MiB` 降至 `6,927 MiB`（节省 `2,324 MiB`）**，工作区估算从 `3,885 MiB` 修正至 `2,894 MiB`。

### 2.6 MoE EP Prims-TS 分离层与 DeepSeek FP8 发送端量化、Cake W4A16 门控及编译器修复（`9eb51e64a872`, `a2b15151fe59`, `714cbf150d90`, `8afd743d38de`）
- **MoE EP DeepSeek FP8 发送端量化（`9eb51e64a872`）**：`MoEEpSplitLayer` 支持 `PrimsTsConfig` 并在发送端将 7168 维 BF16 Token（`14,336 B`）按 `1x128` 块量化为 DeepSeek FP8（`7,168 B` 数值 + `56 × 4 B = 224 B` FP32 缩放 = `7,392 B`），单 Token 跨节点分发载荷减少 **`48.44%`**。
- **Cake W4A16 CUDA 13.0 门控（`a2b15151fe59`）**：显式校验 `CAKE_W4A16_MIN_CUDA = (13, 0)`，防止旧工具链编译失败。
- **TileLang `T.Parallel` `IntImm` 转换（`714cbf150d90`）与 Triton AMD `gfx1250-strict` Block32 硬件上采样（`8afd743d38de`）**：TileLang 通过 `tirx.convert(coalesced_width)` 消除 TVM FFI 裸 Python `int` 断言崩溃；Triton 在 `gfx1250-strict` 校验 lane `n` 与 `n+16` 共享缩放时启用 `v_cvt_scale_pk8` Block32 硬件指令。

## 3. 确定性实验结果汇总

运行 `python experiments/flashinfer_cake_mla_sampling_vsa_attnres_gdn_tilelang_triton_phase128/run.py`：

| 子系统 / 提交 | 核心指标 | 优化前 / 基线 | 优化后 / 新路径 | 增益 / 架构收益 |
|---|---|---:|---:|---|
| Cake NVFP4 MLA (`949a9f2d53ca`) | 短单元解码延迟 | 主干基线 | Setup 阶段预取 `2*cluster` | B200 **`1.026x`** / GB200 **`1.029x`** |
| Cake Sampling R17 (`3812cddafdb7`) | RTX PRO 6000 / B200 采样 | 双核启动 / 常规流 | 8192 紧凑列表 / 整 CTA 尾 | RTX PRO 6000 **`1.413x`** / B200 **`1.125x`** |
| Cake VSA SM90 (`279f9a082fb2`) | Sage-FP8 GQA 主循环指令 | `675` 条 | `633` 条（`cta_loop_iters>=14`） | 路由形状 geomean **`1.0287x`**（峰值 `1.0361x`） |
| Cake AttnRes (`d0ac3f0b1649`) | `M=192, K=6..8` 寄存器与 TMEM | 232 regs / 256 TMEM 列（Spill） | 200 regs / 512 TMEM 列（0 Spill） | B200 & GB300 最高 **`+9.9%`** |
| Cake GDN Train (`025f5acf562f`) | 队列后准备耗时 / 反向峰值显存 | `147.0 ms`（24 sync）/ `9,251 MiB` | `5.8 ms`（0 sync）/ `6,927 MiB` | 准备 **`25.34x`**，节省 **`2,324 MiB`** HBM |
| MoE EP Split (`9eb51e64a872`) | 7168 维单 Token 分发字节 | `14,336 B` (BF16) | `7,392 B` (DeepSeek FP8) | 跨卡 All-to-All 载荷 **`-48.44%`** |
