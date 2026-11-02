# GPU 软件栈：FlashInfer MLA FP8 Page-64 驻留上限与 SM107a 208-SM mHC 链打包、零梯度 Log-Prob 跳过、CuTe GB300/SM100 调度及 TileLang Ascend/SM12x 优化（Phase 180）

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-mla-fp8-gdn-sm107a-mhc-fa-gb300-triton-tilelang-phase180.md)** | **[English](gpu-software-stack-flashinfer-mla-fp8-gdn-sm107a-mhc-fa-gb300-triton-tilelang-phase180.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 180（`priority: 189`）审计了 Blackwell（`sm_100a`）、Blackwell Ultra（`sm_103a` / GB300）、Rubin（`sm_107a` / R200）、RTX Blackwell（`sm_120`）与华为 Ascend 950 算子及编译器栈的 **11 个上游主干提交**：

1. **FlashInfer `cake_trtllm_mla_blackwell` FP8 Page-64 归约器动态 SMEM 驻留上限与 `cake_gdn_det` B200 多轮分裂扫描反向重叠**：
   - `022f245cbea9` (#8343) 在 DeepSeek MLA 解码后端 `cake_trtllm_mla_blackwell`（128 查询头、512 潜在 KV 秩 + 64 RoPE）的 FP8 e4m3 page-64 split-KV 归约器内核中增加 `96 KiB` 动态共享内存预留（内核代码与参数不变），将每 SM 驻留的归约器 CTA 从 `3` 个限制为 `2` 个，在 B200（`sm_100a`）与 GB300（`sm_103a`）上实现最高 **`1.088x`** 加速（B200 `16.224 us -> 14.912 us`，GB300 `15.360 us -> 14.112 us`）。
   - `e53341196bde` (#8342) 在 `cake_gdn_det_train` 中将 B200（`sm_100a`，148 SM）上的融合 BF16-`dbeta`（`HV = 2H`）反向尾部侧流重叠扩展至多轮分裂伴随扫描（split adjoint scan）的第二遍扫描（当总 `tokens * HV >= 393,216` 即 $3 \times 2^{17}$ 时生效），在 $7 \times 4.3\text{k}$ 真实打包序列（`H/HV = 8/16`）上将正向+反向延迟从 `1.1118 ms` 降至 `1.0731 ms`（**`1.036x`** 加速，`0.0 ULP` 逐比特一致）。
2. **FlashInfer `cake_deepgemm_mega_mhc` Rubin `sm_107a` 208-SM 链打包与 `cake_lm_head_loss` 零梯度行跳过**：
   - `c258c2dd23ae` (#8341) 为 `flashinfer.mega_mhc` 新增 SM 数量路由键（区分 `208-SM` 与 `216-SM` R200 `sm_107a` 设备）及 FP8 group-128 UE8M0 链打包（chain-packing）程序（将第 5 轮系数链按每 CTA 4 个 token 打包），在 208-SM R200（`sm_107a`）$T \in [929, 968]$ 区间实现最高 **`1.104x`** 加速（`19.248 us -> 17.440 us`）。
   - `2757266b2a42` (#8268) 在 `cake_lm_head_loss` 非确定性对数概率（log-prob）反向传播中，当 $T \ge 4097$ 时在设备端将 `dlogp != 0` 的有效行紧凑前置并在设备端计数处截断 GEMM（零 Host 回读、完全兼容 CUDA Graph），在 50% 零梯度行（如 RLHF/GRPO 掩码 token，$T=65031$）下实现最高 **`1.943x`** 反向加速（B200 `234.992 ms -> 120.949 ms`，GB300 `1.926x` `190.070 ms -> 98.687 ms`，较编译版 Cut Cross-Entropy 快 `1.292x`）。
   - `33ddcbacf7b7` (#8339) 在 `cake_dsa_indexer_det` 中新增 `l6_solo_unsampled_max_queries = 16384` 策略，将 `sm_100a`/`sm_103a` 上查询数 $\le 16384$ 的非采样调用路由至非配对 `l6` 扫描程序，在 GB300 `Z2cp`（CP16 zigzag）上实现 **`1.095x`** 加速（`2.0902 ms -> 1.9081 ms`）。
3. **FlashAttention CuTe GB300 (`SM103`) 前向 CLC 调度与 SM100 `FwdConfig` 共享内存布局显式化**：
   - `33555791fd24` (#2738) 在 GB300（`SM103`）上提升四个 BF16 前向调度区域，将 `packed-varlen H>=24`、均衡 `packed-varlen MHA` 与稠密因果 `B>=32, K=640..2048` 切换至 Cluster Launch Control（`CLC`），在 `packed-varlen H>=24` 上取得 **`1.192x`** 几何平均加速（最高 `1.421x`）。
   - `b7639cbba7a6` (#3056) 将 SM100 前向内核的 `overlap_sO_sQ` 与 `kv_stage`（224 KB SMEM 预算内 `[2, 32]` 级流水）纳入 `FwdConfig`（`select_sm100_smem_layout` 与 `sm100_fwd_smem_bytes`），消除未校验的共享内存布局偏差（`0` 配置失配）。
4. **Triton AMD 批量 WMMA Scale 布局 / Cluster Barrier 约束与 TileLang Ascend FlashMLA / `sm_12x` 99 KiB SMEM 适配**：
   - `bbe38e574262` (#10733) 修复 Triton AMD 批量 WMMA scale 的 `kDim` 线性布局冗余广播，`fbbb964e3e1d` (#12568) 将 `canIntroduceClusterBarrier` 严格限制在内核入口函数（消除 `noinline` 辅助函数内非法 cluster barrier）。
   - `2031b244f308` (#3678) 在 TileLang 中新增面向华为 Ascend 950 的手动调度 FlashMLA 内核（显式复用 AIC/AIV 核流水线支持 BF16 Prefill 与 FP8/FP4 Decode），`96b947db2cf3` (#3702) 将注意力示例动态适配至 `sm_86/89/120`（如 RTX PRO 6000 Blackwell `sm_120`）的 `99 KiB`（`101,376 B`）每块动态共享内存上限。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+---------------------------------------------------------------------------------------------------+
| Phase 180 GPU Software Stack: Blackwell B200/GB300, Rubin R200 (sm_107a), SM120 & Ascend 950      |
+---------------------------------------------------------------------------------------------------+
  [FlashInfer Cake Blackwell/Rubin Kernels (#8343, #8342, #8341, #8268, #8339)]
    |-- MLA FP8 Page-64 Split-KV (#8343) --> 96 KiB Dyn SMEM Reservation -> Cap 2 CTAs/SM (1.088x)
    |-- GDN Det Train B200 (#8342)       --> Split-Scan Pass-2 Tail Overlap (tokens*HV>=393k, 1.036x)
    |-- Mega-mHC R200 sm_107a (#8341)    --> 208-SM Route Key + 4-Tok/CTA Chain-Packing (1.104x)
    |-- LM-Head LogProb Bwd (#8268)      --> On-Device dlogp!=0 Compaction (T>=4097, 1.943x at 50% 0s)
    +-- DSA Indexer Det (#8339)          --> Unpaired l6 Scan for Unsampled T<=16384 (1.095x GB300)
                                      |
                                      v
  [CuTe FlashAttention SM100/SM103 + Triton + TileLang (#2738, #3056, #10733, #12568, #3678, #3702)]
    |-- CuTe GB300 Fwd (#2738)           --> Packed-Varlen H>=24 -> CLC (1.192x geomean, 1.421x max)
    |-- CuTe SM100 FwdConfig (#3056)     --> Explicit overlap_sO_sQ & kv_stage [2..32] in 224 KB SMEM
    |-- Triton Compiler (#10733, #12568) --> AMD Batched WMMA Scale Fix + Kernel-Only Cluster Barrier
    +-- TileLang (#3678, #3702)          --> Ascend 950 AIC/AIV FlashMLA + SM120 99 KiB SMEM Fit
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/flashinfer_mla_fp8_gdn_sm107a_mhc_fa_gb300_triton_tilelang_phase180/run.py`](../../experiments/flashinfer_mla_fp8_gdn_sm107a_mhc_fa_gb300_triton_tilelang_phase180/run.py) `--check` 100% 验证通过：
- `mla_fp8_page64_reducer_residency_speedup = 1.088`（`96 KiB` 动态 SMEM 预留限制 2 CTAs/SM）
- `gdn_b200_split_scan_overlap_speedup = 1.036`（`tokens * HV >= 393,216` 分裂扫描尾部重叠）
- `mhc_sm107a_208sm_chain_pack_speedup = 1.104`（208-SM R200 `sm_107a` $T \in [929, 968]$ 链打包）
- `lm_head_logprob_bwd_zero_grad_skip_speedup = 1.943`（$T=65031$ 50% 零梯度行设备端紧凑跳过）
- `dsa_unpaired_l6_unsampled_speedup = 1.095`（GB300 非采样 $T \le 16384$ 非配对 `l6` 扫描）
- `fa_gb300_packed_varlen_h24_clc_geomean_speedup = 1.192`（CuTe GB300 `SM103` CLC 调度）
- `fa_sm100_unvalidated_smem_layouts = 0`、`triton_helper_fn_cluster_barriers = 0`、`tilelang_sm12x_smem_budget_overflows = 0`

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **RLHF / GRPO 后训练长序列 Log-Prob 反向计算** | 在 B200/GB300 上针对 $T \ge 4097$ 且含 padding/prompt 掩码（`dlogp = 0`）的批次启用 `cake_lm_head_loss` 设备端非零行紧凑跳过（`#8268`，50% 掩码下 `1.943x` 加速且保持 CUDA Graph 捕获）。 | 全非零梯度（0% 掩码）场景若追求极致无分块延迟，编译版 Cut Cross-Entropy（CCE）更快（但需承担完整 $T \times V$ 显存峰值）。 |
| **Rubin R200 (`sm_107a`) 与 Blackwell MLA/DSA 在线推理** | 在 208-SM R200 上启用 SM-count 感知的 `mega_mhc` 链打包路由（`#8341`），并在 B200/GB300 上启用 FP8 page-64 归约器 96 KiB SMEM 驻留限制（`#8343`）与非采样 $T \le 16384$ 非配对 `l6` 扫描（`#8339`）。 | 在 216-SM R200 上强行覆写 $T \in [929, 968]$ 路由，或在 GB200 上对 $>16384$ 查询的多段因果调用使用非配对 `l6` 扫描。 |
| **GB300 (`SM103`) / RTX PRO 6000 (`sm_120`) / Ascend 950 跨架构部署** | 在 GB300 上启用 CuTe `CLC` 变长前向调度（`#2738`），在 `sm_120` 工作站 GPU 上按 `99 KiB` SMEM 上限缩减 Tile 与 Warp 数（`#3702`），并在 Ascend 950 上采用显式 AIC/AIV FlashMLA 调度（`#3678`）。 | 在 `sm_86/89/120` 上硬编码 Hopper 160 KiB 动态 SMEM Tile（将触发 `101376 bytes` 启动失败），或在 Triton `noinline` 辅助函数中插入 Cluster Barrier（`#12568`）。 |
