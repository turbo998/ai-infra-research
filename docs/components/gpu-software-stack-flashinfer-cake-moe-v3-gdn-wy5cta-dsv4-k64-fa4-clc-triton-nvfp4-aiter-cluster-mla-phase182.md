# GPU 软件栈：FlashInfer Cake Mega-MoE v3 单 FC2 任务、5-CTA/SM GDN WY 内核与 SM107a `K=64` DSV4 解码、FlashAttention-4 2CTA MLA 头部填充与 SM100 因果 CLC、Triton NVFP4 共享内存核算与 AITER `gfx1250` 簇内 Split-K MLA（Phase 182）

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-cake-moe-v3-gdn-wy5cta-dsv4-k64-fa4-clc-triton-nvfp4-aiter-cluster-mla-phase182.md)** | **[English](gpu-software-stack-flashinfer-cake-moe-v3-gdn-wy5cta-dsv4-k64-fa4-clc-triton-nvfp4-aiter-cluster-mla-phase182.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 182（`priority: 191`）审计了 Blackwell（`sm_100a` / B200 / GB200）、Blackwell Ultra（`sm_103a` / GB300 / B300）、Rubin（`sm_107a` / R200 / VR200 / GR100）、AMD `gfx1250`（MI455X）与 CDNA4（`gfx950` / MI350X / MI355X）算子与编译器栈的 **13 个上游主干提交**：

1. **FlashInfer `cake_mega_moe_v3` 单 FC2 任务 TP8/EP1 算子、5-CTA/SM 紧凑三角 GDN WY 前向与 KCP 磁带移交**：
   - `1ce8f915659e` (#8669) 优化 GB300（`sm_103a`，152 SM）上 384 专家、top-6、`hidden=5120` 模型的 `cake_mega_moe_v3` `TP8 EP1`（`T=32 / 128 / 512 / 1024`）Rank 程序：每个 FC2 环形任务一次覆盖所认领专家组的全部 `4` 个 N-clusters（每组仅发 `1` 个 FC2 任务而非 `2` 个），在 `T=32`（`91.26 -> 90.36 us`）与 `T=128`（`176.58 -> 174.97 us`）上较旧版 Cake 提升 **`1.010x`**，且在全部 32 个 Rank 形状上比自动调优 `trtllm_fp4_block_scale_routed_moe` 快 **`1.021x–1.174x`**（`TP8 EP1 T=32` 为 `105.88 -> 90.27 us` = **`1.173x`**，`TP8 EP8 T=4096` 为 `209.05 -> 178.14 us` = **`1.174x`**）。
   - `8e7be5e3a832` (#8665) 在 `cake_gdn_train` 中为 B200（`sm_100a`）与 GB300（`sm_103a`）新增每 SM 驻留 `5` 个 CTA 的紧凑下三角前向 WY 程序 `wy_tc2`（前向与激活检查点重计算共用），在 GB300 上相对 cuDNN FROST（chunk 64）取得最高 **`2.013x`** `fwd+bwd` 加速（`1.2787 ms -> 0.6351 ms`），在 R200（`sm_107a`，212 SM）上取得最高 **`2.527x`** `fwd+bwd` 加速（`0.6136 ms -> 0.2428 ms`）。
   - `283abf0d4eab` (#8638) 在 `cake_gdn_train_kcp` 的 `graph="replay"` 激活检查点步骤中将重计算（recompute）生成的磁带直接移交反向传播（跳过二次 `prep` 与状态遍历），在 32K tokens 上实现 B200 **`1.249x`**（`6.050 -> 4.845 ms`）、GB300 **`1.248x`**（`5.598 -> 4.495 ms`）与 R200 **`1.257x`**（`3.554 -> 2.828 ms`）加速。
2. **FlashInfer SM107a `K=64` FP8 DSV4 解码、稠密投影 GEMM 绑定启动与 SM103 跨频段 QKV 交换**：
   - `a8e34e486334` (#8668) 为 VR200（`sm_107a`，208 SM）的 128-head 均匀收集持久化 DeepSeek-V4 解码路径新增单 CTA `M128`、`K=64` FP8 MMA 程序（`rows >= 48` 时替代双 CTA `K=32` 程序），在 `rows=124, top-k 128x (384)` 上实现 **`1.2778x`** 加速（`14.623 -> 11.444 us`），輸出逐比特一致。
   - `02fa872444d6` (#8650) 使 `cake_dense_projection_gemm` 非确定性入口从缓存键直接跳转至已绑定启动（跳过重规划与 DLPack 转换），在 GB200 与 GB300 的宿主受限（host-bound）eager 反向与检查点步上分别提速 **`1.068x–1.136x`**（GB200 `4096->128` 反向 `56.58 -> 50.10 us` = **`1.129x`**）与 **`1.082x–1.129x`**（GB300 `4096->32` 反向 `38.42 -> 34.03 us` = **`1.129x`**）。
   - `7d929987724c` (#8649) 为 `QkvNormRopeExchangeOverlap` 引入按隐藏维度的每目标切分阈值表（`8192 / 16384 / 32768` 行）及 SM103 8-Rank 跨频段（`_xb`，`M >= 4096`）持久化发射计划。
3. **FlashAttention-4 稠密 2CTA MLA 头部填充与 SM100 因果 CLC + Global LPT 默认调度**：
   - `5f7b05a78bf0` (#3118) 允许 `65–127` 个查询头的稠密吸收式 MLA（如 Kimi-K3 TP=1 的 96 头 MQA）通过 `qheads_first_tma_view` 在内核内自动零填充至 128 行 2CTA 簇 Tile，消除 `"needs Q heads per KV head to divide 128"` 异常，在 B200 解码 `batch=512, s_k=256` 上将延迟从 1CTA 的 `152.4 us` 降至 **`68.5 us`（`2.225x`）**，`batch=64, s_k=8192` 从 `273.5 us` 降至 **`169.6 us`（`1.613x`）**。
   - `6d6392ddc66d` (#3062) 在 SM100/SM103 上默认启用稠密因果前向 CLC（Cluster Launch Control）调度，结合延迟负载 Warp 认领（`clc_late_claim`）与每 SM $\le 2.25$ Tile 时的全局最长处理时间优先排序（Global LPT），在带尾部残缺分区的网格上实现 B200 **`1.07x–1.28x`**、B300 **`1.05x–1.23x`**、GB300 **`1.04x–1.41x`** 加速，并在 B200 `hd256` 2CTA（$\ge 32$ 切片）上取得 **`1.006x–1.148x`**（几何平均 `1.044x`）加速。
4. **Triton 解码 NVFP4 共享内存预算核算、TileLang SM103a/SM107a TMA 门控与 ROCm AITER `gfx1250` 簇内 Split-K MLA / `gfx950` Q256 MXFP4**：
   - `90e50ecd9cb4` (#12613) 修正 Blackwell `FP8 x NVFP4` 稠密矩阵乘的共享内存预算（计入解码后的 BF16 权重 Tile 与转换暂存区），消除超出 GB300 `232,448 B` SMEM 上限的 `236,840 B` / `266,256 B` 越界请求，在 GB300 `M=4096, K=8192, N=16384` 上将延迟从 `2,711.89 us` 降至 **`2,057.72 us`（`1.318x`，`-24.1%`）**；`51f4f818afc1` (#12524) 为 AMD `gfx1250` `scaled_dot` 启用 LDS 预取；TileLang `8587866b22c1` (#3772) 在 CuTeDSL TMA 架构门控中接纳 `sm_103a`（B300）与 `sm_107a`（GR100/Rubin）。
   - ROCm AITER `0f61f2b95d0b` (#7390) 在 `gfx1250` MLA v4.1 `a8w8` 稀疏预填充中通过 `(S, 1, 1)` 工作组簇（`S in {2, 4, 8, 16}`）利用 LDS + TDM 在内核内折叠 Split-K 归约，省去独立 `pa_decode_ps_reduce_hip_kernel` 启动，在 `H=64, N=8, topk=1024` 上将耗时从 `15.26 us` 降至 **`8.36 us`（`1.825x`，`-45.2%`）**；`0e072ca50ee7` (#5988) 新增 `gfx950` Q256 融合 MXFP4 预填充流水线（汇编内加权 M128 调度 + `AsmMoeMode` mode 3 后归约），较 FlyDSL 提升 **`5.40%–8.61%`**（全 API 图回放中位 `1,125.49 us`，最大 NRMS `0.00017653`）。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+--------------------------------------------------------------------------------------------------------+
| Phase 182 GPU Software Stack: Blackwell B200/GB300, Rubin R200/VR200 (sm_107a), gfx1250 & CDNA4 gfx950 |
+--------------------------------------------------------------------------------------------------------+
  [FlashInfer Cake Mega-MoE v3, GDN 5-CTA WY & DSV4 K=64 (#8669, #8665, #8638, #8668, #8650, #8649)]
    |-- Cake Mega-MoE v3 (#8669)     --> GB300 sm_103a TP8/EP1 1 FC2 Task/Group (4 N-Clusters, 1.174x vs TRT-LLM)
    |-- Cake GDN 5-CTA WY (#8665)    --> B200/GB300 wy_tc2 5 CTAs/SM (2.013x GB300 / 2.527x R200 vs FROST)
    |-- Cake GDN KCP Tape (#8638)    --> Graph-Replay Recompute Tape Handoff to Backward (1.257x R200 @ 32K)
    |-- Cake DSV4 SM107a (#8668)     --> 128-Head Uniform-Gather K=64 FP8 MMA rows>=48 (1.2778x VR200)
    +-- Bound Proj GEMM (#8650/#8649)--> Key-to-Bound-Launch Eager Bypass (1.129x) + SM103 8-Rank XBand QKV
                                      |
                                      v
  [FlashAttention-4 + Triton + TileLang + ROCm AITER (#3118, #3062, #12613, #12524, #3772, #7390, #5988)]
    |-- FA4 Dense 2CTA MLA (#3118)   --> In-Kernel TMA Pad 65-127 Q-Heads to 128 (2.225x B200 96-Head Decode)
    |-- FA4 SM100 Causal CLC (#3062) --> Default CLC + Late Load Claim + Global LPT <=2.25 Tiles/SM (1.41x GB300)
    |-- Triton & TileLang (#12613)   --> GB300 Decoded NVFP4 SMEM <=232,448B (1.318x) + SM103a/SM107a TMA Gate
    +-- ROCm AITER (#7390, #5988)    --> gfx1250 Cluster Split-K MLA (1.825x) + gfx950 Q256 MXFP4 (+8.61%)
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/flashinfer_cake_moe_v3_gdn_wy5cta_fa4_clc_aiter_cluster_mla_phase182/run.py`](../../experiments/flashinfer_cake_moe_v3_gdn_wy5cta_fa4_clc_aiter_cluster_mla_phase182/run.py) `--check` 100% 验证通过：
- `cake_mega_moe_v3_t32_self_speedup = 1.010`、`cake_mega_moe_v3_vs_trtllm_max_speedup = 1.174`（`1` FC2 task/group）
- `cake_gdn_5cta_gb300_vs_frost_speedup = 2.013`、`cake_gdn_r200_vs_frost_max_speedup = 2.527`、`cake_gdn_kcp_tape_handoff_r200_speedup = 1.257`
- `cake_dsv4_sm107a_k64_max_speedup = 1.2778`（`rows >= 48` 门槛）、`cake_proj_gemm_bound_launch_gb200_speedup = 1.129`
- `fa_mla_2cta_dense_96h_decode_speedup = 2.225`、`fa_sm100_clc_global_lpt_gb300_max_speedup = 1.41`（`<= 2.25` tiles/SM）
- `triton_gb300_fp8xnvfp4_speedup = 1.318`（`232,448 B` SMEM 预算内）、`tilelang_cutedsl_new_tma_archs = 2`
- `aiter_gfx1250_mla_cluster_splitk_speedup = 1.825`、`aiter_gfx950_q256_mxfp4_vs_flydsl_max_gain_pct = 8.61`

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **GB300 / R200 超大规模 MoE 与 GDN 混合架构训练/推理** | 在 GB300 `sm_103a` `TP8 EP1` 解码/小批次预填充中启用 `cake_mega_moe_v3` 单 FC2 任务程序（`#8669`，较 TRT-LLM 快 `1.174x`），在 B200/GB300 GDN 训练中启用 5-CTA/SM `wy_tc2`（`#8665`，`2.013x`）与 KCP 图回放磁带移交（`#8638`，`1.257x`）。 | 在 VR200 (`sm_107a`) DeepSeek-V4 解码 `rows < 48` 时强行启用 `K=64` 均匀收集程序（背靠背短调用比 `K=32` 慢，应保持 `rows >= 48` 门槛）。 |
| **Blackwell 96 头稠密 MLA 与因果注意力调度** | 对 Kimi-K3 TP=1（96 头）等 `65–127` 头稠密吸收式 MLA 启用 FA4 2CTA 内核内头部填充（`#3118`，解码提速 `2.225x`），并对每 SM $\le 2.25$ Tile 的未切分因果前向启用 CLC + Global LPT（`#3062`，GB300 最高 `1.41x`）。 | 对 `< 65` 头的稠密 MQA 强行使用 2CTA 填充（如 3 头比 1CTA 慢 18 倍），或在 $> 2.25$ tiles/SM 的满波次网格上强开 Global LPT（可能损失 `1.8%–8%`）。 |
| **Blackwell `FP8 x NVFP4` 矩阵乘与 AMD `gfx1250`/`gfx950` 稀疏/MoE 预填充** | 在 GB300 上使用修复解码 BF16 权重 Tile 共享内存核算后的 Triton 3.9 `FP8 x NVFP4` 默认配置（`#12613`，`1.318x`），在 `gfx1250` 小批次稀疏 MLA 预填充中启用 `(S,1,1)` 工作组簇内核内 Split-K 折叠（`#7390`，`-45.2%`），在 `gfx950` 8K–32K MoE 预填充启用 Q256 融合 MXFP4（`#5988`）。 | 在非洗牌（unshuffled）或交错权重上强行分派 AITER Q256 MXFP4（必须通过双投影洗牌标志校验并保留两阶段 FP8 回退）。 |
