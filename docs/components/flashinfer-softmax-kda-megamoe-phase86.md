# FlashInfer Blackwell Per-Tile Softmax Turnover Ring、FlashKDA 10-Slot 单拥有者屏障环与 MegaMoE CuTe DSL Next 深度解析（Phase 86）

> **Language / 语言**: **中文 (当前)** | [English](flashinfer-softmax-kda-megamoe-phase86.en.md)

## 1. 背景与架构演进

在 Blackwell (`sm_100a` B200 / `sm_103a` GB300 / `sm_120a` RTX 5090 & RTX PRO 6000 Blackwell) 与 Hopper (`sm_90a`) 的生产级算子工程中，`flashinfer-ai/flashinfer` 在 2026-10-25 集中合入了 10 项针对长词表 Persistent Cluster Softmax、FlashKDA N16 多阶段辅助 MMA 屏障协议、GB300 TMA 输出写回节流（Store Pacing）、MegaMoE CuTe DSL Next 后端以及 Frost 低延迟 FP8/Block-FP8 Decode 的关键提交：

1. **Blackwell Persistent Cluster Softmax 逐 Tile 翻转环（Per-Tile Turnover Ring）**（[PR #7359](https://github.com/flashinfer-ai/flashinfer/commit/9e016bdfc39624c1659a10d60ed8c9af4c3cea61)）：在 8-CTA 与 16-CTA 分布式共享内存（DSMEM）持久化 Softmax 中消除行间全 CTA 屏障，在写出第 $i$ 行某 Tile 的同时立即将第 $i+1$ 行的对应 Tile 装入同一组寄存器。
2. **FlashKDA N16 10-Slot 单拥有者辅助屏障环（One-Owner Ring of Ten Slots）**（[PR #7358](https://github.com/flashinfer-ai/flashinfer/commit/62c2ada6a5ef277acdd1d744016b335bb9f3f235)）：修复 `kernel_flashkda_bf16_fused_m128` 与 `prepare_bf16_kda_prefill` 在单序列 $\ge 6$ 个 16-token chunk 时因双辅助 MMA Warp 交替复用奇数 5-stage 环导致的 mbarrier 相位（parity）提前一圈放行与死锁。
3. **GB300 (`sm_103a`) 5120 列 Input-Gradient (`dgrad`) `_sw25` 输出写回节流（Paced Output-Store）**（[PR #7361](https://github.com/flashinfer-ai/flashinfer/commit/7dbb267ae89d9c2f6586ded623ad9d10f5f8a303)）：将 `128x256` CTA-pair TMA-store `pd=2` 排空阶段的 8-Warp 突发写入拆解为 7 步各 25 次依赖型 `u32` 乘加（MAD）斜坡，消除 L2/HBM 写拥塞。
4. **Blackwell MegaMoE CuTe DSL Next 标准后端与 Frost FP8 / Router GEMM / MiniMax-H3 SM120 流水线**（[PR #7001](https://github.com/flashinfer-ai/flashinfer/commit/a70a503844d94a07fea1f43ab21dab74476d95d4)、[PR #7212](https://github.com/flashinfer-ai/flashinfer/commit/d3d7c2488783d85e678bd693318bbee19e2fbb97)、[PR #7311](https://github.com/flashinfer-ai/flashinfer/commit/2866b59272d7976d10409e34cf15ce9dd9400f70)、[PR #7310](https://github.com/flashinfer-ai/flashinfer/commit/06da6cc640a059b95286d275ee464e0d26c9c16f)、[PR #7289](https://github.com/flashinfer-ai/flashinfer/commit/ba12a832238575233f2cee795bd93479494fc69a)、[PR #7285](https://github.com/flashinfer-ai/flashinfer/commit/dc39352926e8db106670d138c8cfb53f09e1e388)、[PR #7287](https://github.com/flashinfer-ai/flashinfer/commit/7b220a67a48606c82105cb4cafa22189489583ee)）。

---

## 2. 第一手源码级实现路径与数学公式分析

### 2.1 Blackwell Persistent Softmax 逐 Tile 翻转环（PR #7359）

在 `csrc/cake_blackwell_softmax.cu`、`csrc/cake_blackwell_softmax_cached_cluster.cu` 与 `csrc/cake_blackwell_softmax_cached_cluster_persistent.cu` 中：

- **旧实现瓶颈**：当 `num_tiles` 落在寄存器驻留上界 `kMaxRegTiles = 4` 与共享内存驻留上界 `kMaxSharedTiles = 16` 之间时，Persistent 8-CTA 与 16-CTA Cluster 内核先把第 0 行的所有 Tile 读入 SMEM，再在每一行的主循环末尾执行全 CTA 同步屏障，等待下一行填满 SMEM。
- **逐 Tile 翻转环（Per-Tile Turnover Ring）**：
  - 在写出第 $i$ 行第 $t$ 个归一化输出 Tile 的同时，同一个线程直接从全局内存发起对第 $i+1$ 行第 $t$ 个 Tile 的预取并写入同一组寄存器，随后写回 SMEM 供第 $i+1$ 行归约与归一化使用，**无需任何行间全 CTA 屏障**（唯一例外是在 `sm_100a` 上 16-CTA Cluster 单 Cluster 行数 $>24$ 时回退到逐行填充以规避长驻留发射压力）。
  - Warp 内最大值归约改用硬件指令 `redux.sync.max`（通过位重解释将 IEEE-754 浮点数映射为保序有符号整数后归约）。
  - 调度启发式 `select_cached_launch` 同步更新：
    - `kSharedTilesAnyRowsMaxTiles = 5`（`num_tiles <= 5` 无条件启用 Shared-Tiles）；
    - `kPersistentTripFillNum = 4, kPersistentTripFillDen = 5`（即持久波次填充率 $\ge 0.80$ 时启用 Persistent Cluster）；
    - `kSharedTilesRowsPerClusterFullChunkC16 = 8`、`kSharedTilesRowsPerClusterFullChunkC8 = 7`。
- **实测收益**：在 41 个测试形状上，B200 (`sm_100a`) 几何平均提速 **1.047x**，GB300 (`sm_103a`) 几何平均提速 **1.042x**；其中长词表/长上下文形状 `[128, 128000]` 在 B200 上从 `22.54 us` 降至 `18.08 us`（**1.247x**），在 GB300 上从 `21.33 us` 降至 `17.82 us`（**1.197x**）；`[64, 256000]` 在 B200 上从 `24.90 us` 降至 `20.67 us`（**1.205x**）。

### 2.2 FlashKDA N16 10-Slot 单拥有者辅助屏障环（PR #7358）

在 `csrc/cake_flashkda.cu` 的 `kernel_flashkda_bf16_fused_m128` 与 `prepare_bf16_kda_prefill` 中：

- **根因诊断（跨 Warp 交替拥有奇数槽位引发的相位污染）**：
  - FlashKDA N16 使用两个辅助 MMA Warp（`aux_idx` $\in \{0, 1\}$）按 `chunk = step * 2 + aux_idx` 交替处理 16-token chunk。
  - 原实现将 `aux_pairwise_inputs_ready`、`aux_pairwise_consumed` 与 `aux_inverse_ready` 放在长度为 $S=5$（`NUM_CHUNK_STAGES = 5`）的环上：
    $$\text{stage} = \text{chunk} \bmod 5, \qquad \text{parity} = \left\lfloor \frac{\text{chunk}}{5} \right\rfloor \bmod 2$$
  - 由于 $\gcd(2, 5) = 1$，同一槽位 `stage = 0` 在 `chunk = 0`（`lap = 0, parity = 0`）由 **Warp 0** 消费，而在 `chunk = 5`（`lap = 1, parity = 1`）却由 **Warp 1** 消费！当 Warp 1 在 `chunk = 1` 慢于 Warp 0 的 `chunk = 0, 2, 4` 时，Warp 0 推进 `stage = 0` 的完成计数会使 Warp 1 在等待更早周期时看到翻转后的 parity 并**提前一整圈放行**，造成偶发 CTA 死锁或数值损坏。
- **10-Slot 单拥有者环修复**：
  - 引入 `NUM_AUX_RING_STAGES = 2 * NUM_CHUNK_STAGES = 10`：
    $$\text{aux\_slot} = \text{chunk} \bmod 10, \qquad \text{aux\_parity} = \left\lfloor \frac{\text{chunk}}{10} \right\rfloor \bmod 2$$
  - 因为 $10 \equiv 0 \pmod 2$，偶数槽位 $\{0, 2, 4, 6, 8\}$ 永远属于 `aux_idx = 0`，奇数槽位 $\{1, 3, 5, 7, 9\}$ 永远属于 `aux_idx = 1`（跨 Warp 冲突次数从 20 个 chunk 中的 `15` 次严格降为 `0` 次）。
  - 共享内存头部的 mbarrier 数量从 `97 -> 112`（N16）和 `107 -> 122`（prepared 路径），仍完全容纳在既有 `1024 B` 对齐头部内，寄存器占用保持 `64 regs` 不变。在 B200 与 GB300 上连续执行 **200,000** 次 N16 与 **1,000,000** 次 prepared 启动实现 **0 挂起、0 错数**。

### 2.3 SM103a `_sw25` 输出写回节流、MegaMoE CuTe DSL Next 与 Frost/Router/MiniMax 优化

| 提交 / PR | 核心改动 | 关键定量收益 |
|---|---|---|
| [PR #7361](https://github.com/flashinfer-ai/flashinfer/commit/7dbb267ae89d9c2f6586ded623ad9d10f5f8a303) (`7dbb267`) | 在 `sm_103a` (GB300) `5120->512` 与 `5120->1024` K-major `dgrad` 的 `128x256` CTA-pair TMA-store `pd=2` 排空阶段，每个 Epilogue Warp 按步长 $n=25$ 次依赖型 `u32` MAD 延迟发射，将 8-Warp 突发 TMA Store 转化为 7 步斜坡 | GB300 $T=16384$ `5120->512` `54.3 -> 52.7 us` (**1.030x**)，`5120->1024` `92.1 -> 91.1 us` (**1.011x**) |
| [PR #7001](https://github.com/flashinfer-ai/flashinfer/commit/a70a503844d94a07fea1f43ab21dab74476d95d4) (`a70a503`) | 新增 `sm100_nvfp4_nvfp4_bf16_cutedsl_next` 与 `sm100_mxfp8_mxfp8_bf16_cutedsl_next`（46 个导出 Python 模块），支持 `swiglu` (`alpha=1.702, beta=1.0`)、`situ`、FC2 前路由权重缩放与 `FinalizeMode::CAKE` | 统一 Blackwell MegaMoE NVFP4/MXFP8 AOT 导出管线 |
| [PR #7212](https://github.com/flashinfer-ai/flashinfer/commit/d3d7c2488783d85e678bd693318bbee19e2fbb97) (`d3d7c24`) | `FrostLowLatencyRunner` (`H=2048, I=512, top_k=10`) 扩展 SM90 (`wgmma`) 与 SM120 (`warp_mma`) grouped FP8 / DeepSeek $128\times 128$ block-FP8，限制原生 FP8 累加跨度 $\le K64$ 后转 FP32 累加 | 消除 SM90/SM120 FP8 尾数漂移并淘汰 Triton 解量化 fallback |
| [PR #7311](https://github.com/flashinfer-ai/flashinfer/commit/2866b59272d7976d10409e34cf15ce9dd9400f70) (`2866b59`) | `cake_blackwell_bf16_fp4` `tcgen05` `cta_group::2` M128 pair 将 `B_descale` 预取移至 Warp 7，翻倍 `bsrc_done`/`smem_b_full`/`sf_done` mbarrier，并将 TMEM 分配后 `barrier.sync.aligned 7` 从 512 线程收窄至 160 线程 | 降低 SM100/SM103 2-CTA M128 同步开销 |
| [PR #7310](https://github.com/flashinfer-ai/flashinfer/commit/06da6cc640a059b95286d275ee464e0d26c9c16f) (`06da6cc`) | `kimi_k3_tp12_tail` 将 `GROUPED_MAX_TOKENS` 从 `256` 提升至 `960`，使 $M \in [256, 960]$ 走 `("k1_twoshot_ess:grouped", "k3_ess:grouped")` | GB300 $M=512$ 从 `60.59 -> 56.10 us` (**1.080x**)，$M=256$ **1.060x** |
| [PR #7289](https://github.com/flashinfer-ai/flashinfer/commit/ba12a832238575233f2cee795bd93479494fc69a) (`ba12a83`) | `minimax_h3_sm120_varlen_attention_fp8` 改为 `KV_STATS` + `QUANTIZE` + 单位缩放 `mma.sync kind::f8f6f4` 分离 K/V 重填三阶段流水线 | RTX 5090 `center_4s` 从 `73,983 -> 52,679 us` (**1.404x**) |
| [PR #7285](https://github.com/flashinfer-ai/flashinfer/commit/dc39352926e8db106670d138c8cfb53f09e1e388) (`dc39352`) | `cake_fused_norm_combine` `OWNER_PIPELINED` ($\text{tokens} \ge 480$) 将流水滞后从 `2 * num_bids` 增至 `3 * num_bids`、最终写回滞后 4 步 | 8x B200 `30.656 -> 30.112 us` (**1.018x**)，8x B300 **1.016x** |
| [PR #7287](https://github.com/flashinfer-ai/flashinfer/commit/7b220a67a48606c82105cb4cafa22189489583ee) (`7b220a6`) | `cake_router_gemm` 为 $M \in [1, 16] \times K \in \{6144, 7168\}$ 生成 32 个专属 shape 调度，并在 $M \in [7, 16], K=7168$ 提供 `narrow` 1-CTA/SM 变体 | SM100/SM103 $K=6144, E=256, M=16$ 从 `6.400 -> 4.320 us` (**1.481x**) |

---

## 3. 可复现实验验证（`experiments/flashinfer_softmax_kda_megamoe_phase86`）

运行 `python experiments/flashinfer_softmax_kda_megamoe_phase86/run.py --check` 通过全部确定性校验：
- **Softmax Turnover Ring**：B200 `[128, 128000]` 加速比 `1.247x`，GB300 `1.197x`；
- **FlashKDA 10-Slot Ring**：20 个连续 chunk 上 5-slot 跨 Warp 冲突 `15` 次 vs 10-slot `0` 次；
- **SM103a Paced Store & Router GEMM**：8-store 突发转为 7 步 $\times 25$ `u32` MAD 斜坡；Router GEMM $M=16$ 加速比 `1.481x`。

## 4. 架构师设计练习与可证伪问题

1. **奇数槽位环跨 Warp 相位校验练习**：当 $W$ 个消费者 Warp 轮流消费深度为 $S$ 的 mbarrier 环时，证明 $\gcd(W, S) = W$（即 $S$ 是 $W$ 的整数倍）是每个物理槽位在任意圈数上仅由唯一 Warp 拥有的充要条件。
2. **可证伪假设**：在 `sm_100a` 16-CTA Cluster Softmax 中，若对 $>24$ 行的 chunk 强行开启逐 Tile 翻转环而非逐行重填，是否会因未完成全局加载队列占满 MIO 发射槽而导致总耗时反而劣于逐行重填？
