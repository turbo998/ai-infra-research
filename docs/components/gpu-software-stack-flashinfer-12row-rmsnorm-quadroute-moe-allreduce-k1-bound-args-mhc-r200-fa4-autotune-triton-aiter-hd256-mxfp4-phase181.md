# GPU 软件栈：FlashInfer 12-Row RMSNorm-SiLU-Gate 短前向、Quad-Route MoE AllReduce、K1 Bound-Args 稠密投影、R200 `sm_107a` mHC 双布局、CuTe GB300 子进程自调优与 AITER `gfx950` `hd256` MXFP4/6（Phase 181）

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-12row-rmsnorm-quadroute-moe-allreduce-k1-bound-args-mhc-r200-fa4-autotune-triton-aiter-hd256-mxfp4-phase181.md)** | **[English](gpu-software-stack-flashinfer-12row-rmsnorm-quadroute-moe-allreduce-k1-bound-args-mhc-r200-fa4-autotune-triton-aiter-hd256-mxfp4-phase181.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 181（`priority: 190`）审计了 Blackwell（`sm_100a` / B200）、Blackwell Ultra（`sm_103a` / GB300 / B300）、Rubin（`sm_107a` / R200）、AMD CDNA4（`gfx950` / MI350X）、`gfx1250` 与华为 Ascend NPU 算子及编译器栈的 **12 个上游主干提交**：

1. **FlashInfer `cake_rmsnorm_silu_gate_train` 12-Rows/SM 短前向、`cake_moe_finalize_allreduce_fusion` WS2 Quad-Route 与 `cake_dense_projection_gemm` K1 绑定参数列表**：
   - `1798b527fa18` (#8505) 将 `cake_rmsnorm_silu_gate_train` 的 16-threads/row 短前向程序（16 rows/CTA）在 SM100（B200，148 SM）与 SM103（GB300，152 SM）上的分发上限从 `10 rows/SM` 提升至 `12 rows/SM`，将单波次 `111 CTAs`（B200，$1480 < T \times H \le 1776$）与 `114 CTAs`（GB300，$1520 < T \times H \le 1824$）保留在 16-threads/row 短前向而非回退至 32-threads/row 前向（`182–228 CTAs`），在 B200 上实现 **`1.259x–1.312x`** 前向加速（`800x2x4096` 从 `8.32 us` 降至 `6.34 us`），在 GB300 上实现 **`1.250x–1.304x`** 前向加速（`832x2x4096` 从 `8.20 us` 降至 `6.29 us`），激活重计算整步加速 **`1.128x–1.154x`**。
   - `283300b79da4` (#8503) 在 `cake_moe_finalize_allreduce_fusion` 中为 `world_size = 2`、`tokens <= 142` 的小批次（`top_k = 8, hidden_dim = 7168`，`sm_100a`/`sm_103a`）新增 4 个 quad-route 程序（每轮处理 4 条 expert routes 而非 8 条，寄存器上限允许每 SM 驻留 4 个 CTA，将最多 142 个 2-CTA clusters 放入单个驻留波次），在 B300 上实现 **`1.041x`** 几何平均加速（`T=128` 最高 **`1.070x`**，`15.81 us -> 14.77 us`），在 B200 上实现 **`1.043x`** 几何平均加速（`T=128` 最高 **`1.069x`**，`14.72 us -> 13.66 us`）。
   - `27b38c15c3a7` (#8456) 在 `cake_dense_projection_gemm`（`sm_100a`/`sm_103a`/`sm_107a`）中为非确定性 eager K1 投影调用缓存有序内核参数绑定列表（`bound_args`）与 PDL 标志，消除每次调用的 `_resolve_ordered_args` 字典查找与列表分配，将 Host 端启动耗时从 `36.0–38.8 us` 降至 `19.1–23.8 us`，在 GB300 上实现 **`1.602x–1.684x`** `2fwd+bwd` 加速（`(256, 3072, 1536)` 从 `156.4 us` 降至 `92.9 us`），在 GB200 上实现 **`1.593x–1.673x`** 加速（`152.7 us -> 91.3 us`）。
2. **FlashInfer `cake_mhc_det` Rubin R200 (`sm_107a`) 双布局 `fn` 分裂与 `cake_blackwell_softmax` 双波次低延迟内核**：
   - `dc4bf3420d01` (#8488) 将 `cake_mhc_det_train` 激活检查点重计算路径上的 256 线程双布局 `fn` 分裂内核（单次启动同时写出正向与转置 `fn` 布局，供反向传播直接复用而无需二次分裂）扩展至 216-SM NVIDIA R200（`sm_107a`，$T \le 4096, \text{hidden}=4096, 4\text{ streams}$），每个检查点步减少 1 次内核启动，在 R200 上取得 **`1.017x–1.033x`** 路径加速（`T=256` 从 `44.01 us` 降至 `42.62 us`，`T=2048` 从 `99.51 us` 降至 `96.59 us`）。
   - `936cd32a539a` (#8449) 在 `cake_blackwell_softmax` 中新增仅在 $\le 2$ 个驻留 cluster 波次时启用的低延迟共享 Tile 变体 `c8_t4_s4_lat` 与 `c16_t4_s4_lat`（含 PDL 孪生内核），在保持 `#8236` 逐比特精确数值的同时，在 B200 `32 x 151936` 上实现 **`1.096x`** 加速（`12.032 us -> 10.976 us`），在 GB300 `64 x 151936` 上实现 **`1.035x`** 加速（`17.960 us -> 17.360 us`）。
3. **CuTe FlashAttention 显式 `FwdConfig` 子进程并行自调优与 MLA CTA 内核选择暴露**：
   - `54b044c8cf89` (#2743) 新增显式 `FwdConfig` 自调优边界：在全屏障后探测磁盘缓存，仅对缺失内核启动按 GPU UUID 隔离的并行 fake-tensor 子进程编译（基于 `fwd_codegen_config` 去重，涵盖 `SwapAB` 候选），并在调用方张量上捕获预分配 CUDA Graph 顺序计时、以 `equal_nan` 校验变长未写行，将 GB300 热调优耗时从 **`4.4–6.6 s` 降至 `0.4–0.9 s`**（最高 **`11.0x`** 加速）。
   - `1dc8232fe209` (#3091) 在 `flash_attn_func` 与 `flash_attn_varlen_func` 公共入口暴露 `MlaFwdConfig.kernel`（`2cta` / `1cta` / `1cta_kb64` 共 `3` 种 CTA 候选）并通过 `validate_mla_fwd_config` 拦截非法配置。
4. **Triton 布局/指针/TDM 编译器修复、TileLang Ascend 批量 3D `ND2NZ` 与 ROCm AITER `gfx950` `hd256` MXFP4/6**：
   - `a972119e92a2` (#12572) 在张量布局验证器中拒绝非 AMD 目标上的 `#ttg.generic_linear` 编码（消除 NVIDIA `convert_layout` `ldmatrix` 断言崩溃）；`a94dac3739e8` (#12601) 在 AMD 指针规范化 `getForwardSliceImpl` 中引入 `visited` 集合，消除 32 分支共享体 `scf.if` 内核的 $O(2^B)$ 指数级编译超时（`>30 s`）；`1fad2df0604c` (#12522) 修复 `gfx1250` TDM `mbarrier` 到达计数，确保每个启用屏障的 TDM 操作按每 Warp 精确贡献 1 次到达；TileLang `fa69f7fa817c` (#3689) 支持 Ascend 批量 3D GM-to-L1 `ND2NZ` 拷贝与越界批次保护。
   - ROCm AITER `6a4fa17deb06` (#7173) 新增面向 `gfx950`（CDNA4 / MI350X）的首个 `head_dim = 256`（`hd256`）MXFP4/MXFP6 `mha_v4` 汇编内核（KV Tile 减半至 `64` 以容纳 128 VGPR $O$ 累加器，在 `[b, s, 2h, 128]` 视图上复用 `hd128` 量化生产者与分块对角 Hadamard 变换），在 `s=8192` 上达到 **`3,652 TFLOP/s`**（较 `hd128` `3,082 TFLOP/s` 提升 **`+18.5%` / `1.185x`**，相对 BF16 余弦相似度 `0.980`）；`bc12ddf440f9` (#5578) 新增 `gfx950` Triton/Gluon MXFP8 FlashAttention v2 前向与反向内核。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+------------------------------------------------------------------------------------------------------+
| Phase 181 GPU Software Stack: Blackwell B200/GB300, Rubin R200 (sm_107a), CDNA4 gfx950 & Ascend NPU  |
+------------------------------------------------------------------------------------------------------+
  [FlashInfer Cake Blackwell & Rubin Kernels (#8505, #8503, #8456, #8488, #8449)]
    |-- RMSNorm-SiLU-Gate (#8505)     --> 12 Rows/SM Short Fwd (111/114 CTAs, 1.312x B200 / 1.304x GB300)
    |-- MoE Finalize-AllReduce (#8503)--> WS2 Quad-Route <=142 Tokens (4 CTAs/SM, 1.070x B300 / 1.069x B200)
    |-- Dense Projection GEMM (#8456) --> Cached K1 Bound-Arg Lists + PDL (1.684x GB300 / 1.673x GB200)
    |-- mHC Det Recompute (#8488)     --> R200 sm_107a 216-SM Dual-Layout fn Split T<=4096 (1.033x)
    +-- Blackwell Softmax (#8449)     --> <=2-Wave Shared-Tile c8/c16_t4_s4_lat + PDL (1.096x B200)
                                      |
                                      v
  [CuTe FlashAttention + Triton + TileLang + ROCm AITER (#2743, #3091, #12572, #12601, #12522, #7173)]
    |-- CuTe Fwd Autotune (#2743/#3091)-> Disk-Cache Barrier + Parallel Fake-Tensor + MLA 2cta/1cta (11.0x)
    |-- Triton Compiler (#12572/#12601)-> Non-AMD generic_linear Reject + Visited-Set scf.if + gfx1250 TDM
    |-- TileLang Ascend (#3689)       --> Batched 3D GM-to-L1 ND2NZ Copy + OOB Batch Preservation
    +-- ROCm AITER gfx950 (#7173/#5578)-> hd256 MXFP4/6 ASM (ts_kv=64, 3652 TFLOP/s, +18.5%) + MXFP8 FA2
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/flashinfer_12row_rmsnorm_quadroute_moe_k1_bound_mhc_r200_fa4_autotune_aiter_hd256_phase181/run.py`](../../experiments/flashinfer_12row_rmsnorm_quadroute_moe_k1_bound_mhc_r200_fa4_autotune_aiter_hd256_phase181/run.py) `--check` 100% 验证通过：
- `rmsnorm_silu_gate_12row_b200_fwd_speedup = 1.312`、`rmsnorm_silu_gate_12row_gb300_fwd_speedup = 1.304`（`1,776` / `1,824` 最大行数）
- `moe_finalize_allreduce_ws2_quad_route_max_speedup = 1.070`（WS2 `tokens <= 142` quad-route 单波次驻留）
- `dense_projection_k1_bound_args_gb300_speedup = 1.684`（GB300 K1 绑定参数列表 `2fwd+bwd` 加速）
- `mhc_det_r200_sm107a_dual_fn_speedup = 1.033`、`blackwell_softmax_two_wave_lat_speedup = 1.096`
- `fa_cute_gb300_warm_autotune_speedup = 11.0`（`4.4 s -> 0.4 s`）、`fa_mla_autotune_cta_choices = 3`
- `aiter_gfx950_hd256_s8192_tflops = 3652.0`、`aiter_gfx950_hd256_mxfp4_tflops_speedup = 1.185`

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **Blackwell / Rubin 稠密投影与融合激活重计算** | 在 B200/GB300/R200 小微批次或高 CP/TP 切分（`M <= 512`）下启用 `cake_dense_projection_gemm` K1 绑定参数列表（`#8456`，`1.684x`）与 `12 rows/SM` 短前向（`#8505`，`1.312x`），并在 R200 `sm_107a` $T \le 4096$ 检查点路径启用双布局 `fn` 分裂（`#8488`）。 | 对 `deterministic=True` 调用强行绕过参数校验，或在 $T \times H > 1824$ 的大批次上强行使用 16-threads/row 短前向。 |
| **双卡（WS2）MoE 低延迟解码与小批次采样** | 在 B200/B300 `world_size = 2, tokens <= 142` 解码/小批次预填充中启用 quad-route `moe_finalize_allreduce_fusion`（`#8503`），并在 $\le 2$ 波次词表采样中启用 `c8/c16_t4_s4_lat`（`#8449`，`1.096x`）。 | 在 `tokens > 142` 或 `world_size > 2` 时使用 quad-route（应保持 oct-route 以摊销多轮跨卡同步开销）。 |
| **AMD MI350X (`gfx950`) `hd256` 注意力与 CuTe 生产调优** | 在 MI350X 长序列（`s=4096..16384`）`head_dim=256` 模型上启用 AITER `mha_v4` MXFP4/6 汇编内核（`#7173`，`3,652 TFLOP/s`），并在 GB300 上启用 CuTe 磁盘缓存屏障并行子进程自调优（`#2743`）。 | 在 `hd256` MXFP4 上启用稀疏注意力（`mha_v4` 暂不支持 `hd256` sparse，切勿静默回退至 `hd128` 对象）。 |
