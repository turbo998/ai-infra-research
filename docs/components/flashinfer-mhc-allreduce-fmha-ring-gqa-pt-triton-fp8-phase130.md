# FlashInfer MNNVL mHC 融合 AllReduce、GB300 4-Stage FP8 Context FMHA 环、CuTe DSL 5-Stage 页表环修复与 Triton AMD 打包 FP8e4nv 上转（Phase 130）

> **English Edition**: [flashinfer-mhc-allreduce-fmha-ring-gqa-pt-triton-fp8-phase130.en.md](flashinfer-mhc-allreduce-fmha-ring-gqa-pt-triton-fp8-phase130.en.md)  
> **Phase 130 (`priority: 139`) · 130 阶段 & 400 实验套件双里程碑 · Track 1（GPU 核函数与编译器层）**  
> **证据等级**: `implementation` + `author-benchmark`（上游源码/PR 提交）+ `own-offline`（本项目 CPU 确定性建模，非 GPU 硬件重测）

---

## 1. 背景与核心架构问题

在 Blackwell (`sm_100a` B200 / `sm_103a` GB300 / `sm_120a` RTX 5090 & RTX PRO 6000)、Rubin (`sm_107a` R200)、Hopper (`sm_90a`) 与 AMD RDNA3/CDNA3/4 构成的算子栈中，2026-10-26 至 2026-10-28 的 10 项上游提交揭示了五个深层工程拐点：

1. **多流超连接（mHC）解码边界的内核启动与同步墙**：在 GLM-5.3-Flash（`hidden=4096`，4 条 mHC 残差流，TP8 MNNVL 部署于双 4-GPU GB300 托盘）中，每个 Attention 与 MoE/MLP 子层后都需完成 TP8 All-Reduce、4 流 mHC Post-Map 混合以及下一层 24 个混合 Logits（`4 pre + 4 post + 16 comb`）的 Pre-Norm 投影。单个解码步包含 **89 个子层边界**；在解码批大小 `1..32` 下，若每个边界启动 3 个独立内核（共 267 次启动），延迟完全被启动与跨卡同步吞噬。
2. **GB300 (`sm_103a`) 输入梯度投影 GEMM 与 FP8 Context FMHA 流水线深度分化**：`dense_projection_gemm` 默认采用 10-Warp 流水线，但在 `sm_103a` 的 `kn` 布局输入梯度行（`4096->512`、`5120->512`、`kv_a 6144->576`）上，8-Warp CTA 布局具有更优的寄存器/TMA 平衡；同时 GB300 上精确 FP8 Context FMHA 将 K/V 环从 3-stage 提升至 4-stage，而 B200 保持 3-stage。
3. **SM120 变长扩散注意力生产者-消费者重叠宽度自适应**：MiniMax-H3 FP8 打包变长注意力在 170-SM RTX 5090 与 188-SM RTX PRO 6000 上若固定重叠宽度 `P = 32`，在短序列（受预处理生产者限制）或超长填充序列下无法占满 SM。
4. **CuTe DSL 分页 GQA 解码中的页表环（Page-Table Ring）竞争漏洞**：`GroupedQueryAttentionDecodePaged` 中 TMA K warp 在等待 `tma_order_v_nbar` 前就将 tile `s + 1` 的页索引预取进共享内存页表环。当 `prefetch_iters = 2` 时，TMA V warp 仍可能读取 `s - 3 .. s` 共 `prefetch_iters + 3 = 5` 个槽位，但旧版 `pt_stages = 4` 导致 K warp 超前时将 tile `s` 的 V 页索引覆盖为 tile `s + 4`，造成 30%–50% 的相对误差。
5. **Hopper VSA 双步 Shuffle-Exchange 输出存储与 Triton AMD `fp8e4nv -> fp16` 打包乘法上转**：Hopper 块稀疏 GQA 通过两步蝴蝶交换减少 32 条 `SHFL` 指令；Triton AMD 后端用打包 `i32` 掩码 + 单次 `2^(15-7)` FP16 乘法替代逐元素整数指数加法与 8 项次正规数查找表（LUT），在 RDNA3 GEMM 上带来最高 **3.0x** 加速。

---

## 2. 一手源码证据与机制深挖（10 项上游提交）

| # | 仓库与提交 | 日期 | 核心机制与架构量化指标 |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@1af7b11` (`#7758`) | 2026-10-28 | `cake_dense_projection_gemm`（126 程序 / 123 模板）为 `sm_103a` 新增两个 8-Warp CTA 程序：`dense_proj_gemm_kn_n256_tma1_w8rs`（服务 `4096->512` 与 `kv_a 6144->576`）和 `w8r`（服务 `5120->512`）。在 `T=16231` 下延迟分别从 `42.8 -> 41.2 µs` (**1.040x**)、`52.2 -> 49.7 µs` (**1.050x**)、`67.1 -> 64.0 µs` (**1.048x**)。 |
| 2 | `flashinfer-ai/flashinfer@cb91a3e` (`#7752`) | 2026-10-28 | `flashinfer.cake_fmha`（每目标 28 个程序）在 GB300 (`sm_103a`) 上将两个 FP8 精确程序（heads `32/4`、page `64`）的 K/V 环从 3-stage 升级为 **4-stage**，host scales 提速 **1.081x**，device scales 提速 **1.078x**；同时为 split-KV 引入更小 pack 计划。 |
| 3 | `flashinfer-ai/flashinfer@4c5d951` (`#7756`) | 2026-10-28 | `minimax_h3_sm120_varlen_attention_fp8`（`[tokens, 56, 128]`）按注意力加权平均段长 $L = \sum s_i^2 / \sum s_i$ 动态选择尾部注意力 CTA 数 $P$（生产者占 $2P$ CTA）：当 $L \le 1792$（170 SM）或 $L \le 2304$（188 SM）时取 $P=64$；当 `padded_tokens > 16384` 且 $L \ge 4096$ 时取 $P=64$（170 SM）或 $P=48$（188 SM）；其余保持 $P=32$。 |
| 4 | `flashinfer-ai/flashinfer@3c7e075` (`#6299`) | 2026-10-28 | 修复 CuTe DSL `GroupedQueryAttentionDecodePaged` 页表环竞争：将 `self.pt_stages` 从 `4` 增至 `5` 并强制断言 `pt_stages >= prefetch_iters + 3`，彻底消除 `q_len_per_req > 1` 且单 CTA 迭代 $\ge 16$ 时 tile `s` 误读 tile `s + 4` V 页导致的 **30%–50%** 间歇性误差；同时修复 `reduction="none"` 下 attention sinks 归约。 |
| 5 | `flashinfer-ai/flashinfer@3957698` (`#7695`) | 2026-10-28 | `cake_grouped_fp8_gemm` 将小块 `UE8M0` swap-AB 程序 `bs_ue8m0_swap_rp32_s11` 与 `bs_ue8m0_swap_c1` 补齐至 `sm_100a`，使全部 **30 个程序**（28 条 host 路由：14 条 FP32 scale + 14 条 UE8M0 block scale）在 `sm_100a` 与 `sm_103a` 上完整对齐。 |
| 6 | `flashinfer-ai/flashinfer@ae13e9c` (`#7682`) | 2026-10-28 | Hopper (`sm_90a`) `cake_vsa_sm90` 更新 `h16-m8192-n8192-k5`（$g \in \{4, 8\}$）紧凑 GQA 程序：输出 tile 采用两步 `shfl.sync.bfly` 交换（每轮 2 次预选 + 2 次蝴蝶洗牌 + 4 次后选），使 `SHFL` 指令数从 `97 -> 65`（BF16，**-32.99%**）和 `90 -> 58`（Sage-FP8，**-35.56%**）。 |
| 7 | `flashinfer-ai/flashinfer@499012e` (`#7694`) | 2026-10-28 | `cake_muse_spark_norm`（hidden `8192`）配置数从 `68 -> 77`（新增 GB300 quarter-row next-RMSNorm 及 R200 `M > 4096` row-ring 路由）；`cake_tp_moe_finalize_rmsnorm`（hidden `4096`、top-k `8`）配置数从 `20 -> 22`。 |
| 8 | `flashinfer-ai/flashinfer@866313f` (`#6797`) | 2026-10-28 | 新增实验性融合通信算子 `flashinfer.comm.allreduce_mhc_post_prenorm`：在 Blackwell MNNVL TP8 上单次启动完成 one-shot All-Reduce、4 流 mHC Post-Map 与 24-logit Pre-Norm 部分和，将 GLM-5.3-Flash 解码步的 89 个子层边界从每边界 3 个内核压缩至 2 个（单步节省 **89 次内核启动**，`267 -> 178`）。 |
| 9 | `triton-lang/triton@d3229a1` (`#12518`) | 2026-10-27 | 在 `ConvertWarpSpecializeToLLVM.cpp` 中回退单 Warp 分区提前将 `ttg.warp_id` 折叠为常量 `0` 的逻辑，保留 `ttg.warp_id` 交由 `NVGPUToLLVM` 处理相对 Warp ID。 |
| 10 | `triton-lang/triton@7e0b1a0` (`#12492`) | 2026-10-26 | 优化 AMD RDNA3/CDNA2/3 软件 `fp8e4nv -> fp16` 上转：将两个 FP8 值打包进单个 `i32` 进行符号/尾数掩码，并乘以 $2^{15-7} = 256.0$ 一步完成指数重偏置与次正规数归一化，移除 8 项 select LUT，在 RDNA3 部分 GEMM 上提速达 **3.0x**。 |

---

## 3. 关键数学公式与确定性实验验证

### 3.1 CuTe DSL 分页 GQA 页表环槽位下界不变量

在 `GroupedQueryAttentionDecodePaged` 中，设 K 预取深度为 $D_{\text{pf}} = \text{prefetch\_iters}$。在迭代 $s$ 顶部，TMA K warp 尚未等待 `tma_order_v_nbar` 即向环槽位 $(s + 1) \bmod N_{\text{pt}}$ 写入 tile $s+1$ 的页索引，而此时 TMA V warp 可能仍在消费迭代 $s - D_{\text{pf}} - 1, \dots, s$ 的槽位。因此同时活跃的环槽位数为：

$$N_{\text{live}} = \bigl(s + 1\bigr) - \bigl(s - D_{\text{pf}} - 1\bigr) + 1 = D_{\text{pf}} + 3$$

当 $D_{\text{pf}} = 2$ 时，$N_{\text{live}} = 5$。若 $N_{\text{pt}} = 4 < 5$，槽位 $(s + 4) \bmod 4 \equiv s \bmod 4$ 发生别名覆盖，导致 tile $s$ 的 V 张量从 tile $s + 4$ 的物理页加载。升级为 $N_{\text{pt}} = 5 \ge D_{\text{pf}} + 3$ 后竞争彻底消除。

### 3.2 SM120 MiniMax-H3 注意力加权段长与重叠宽度公式

给定打包序列段长集合 $\{s_1, s_2, \dots, s_m\}$，每个预处理 token 对应的注意力计算量正比于加权平均段长：

$$L = \frac{\sum_{i=1}^{m} s_i^2}{\sum_{i=1}^{m} s_i}$$

我们的确定性实验套件 `experiments/flashinfer_mhc_allreduce_fmha_ring_gqa_pt_triton_fp8_phase130/run.py` 验证了全部 5 项不变量：
- `sm103a_w8_all_rows_faster = true`（`1.040x` / `1.050x` / `1.048x`）
- `gb300_4stage_fp8_faster_than_3stage = true`（`1.081x` host / `1.078x` device）
- `cute_dsl_5stage_pt_ring_eliminates_race = true`（`required_live_slots = 5`）
- `vsa_sm90_shfl_reduced_by_32_instructions = true`（BF16 `97 -> 65`，Sage-FP8 `90 -> 58`）
- `glm53_mhc_saves_89_launches_per_decode_step = true`（`267 -> 178` launches/step）

---

## 4. 解决方案架构师决策指南与可证伪问题

- **MNNVL 多流残差模型部署**：对于采用 mHC（Manifold-Constrained Hyper-Connections，如 GLM-5.3-Flash 4 流残差）的模型，在 Blackwell MNNVL TP8 上应优先启用 `allreduce_mhc_post_prenorm` 将跨卡 All-Reduce 与流混合融合，避免每步 89 次边界同步放大尾延迟。
- **可证伪后续问题**：当 GLM-5.3-Flash 解码批大小从 `B=32` 增大至 `B=256` 时，MNNVL one-shot All-Reduce 切换为 two-shot 归约后，融合 mHC Post-Map + Pre-Norm 的寄存器压力是否会触发 CTA 驻留率下降并产生融合收益反转点？
