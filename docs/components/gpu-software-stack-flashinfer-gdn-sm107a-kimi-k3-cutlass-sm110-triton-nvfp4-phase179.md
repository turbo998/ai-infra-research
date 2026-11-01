# GPU 软件栈：FlashInfer SM107a GDN 环形状态修正、B200/B300 Kimi-K3 持久令牌队列、CUTLASS Thor SM110 与 Triton NVFP4 行缩放融合（Phase 179）

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-gdn-sm107a-kimi-k3-cutlass-sm110-triton-nvfp4-phase179.md)** | **[English](gpu-software-stack-flashinfer-gdn-sm107a-kimi-k3-cutlass-sm110-triton-nvfp4-phase179.en.md)**

---

## 1. 核心架构问题与源码证据（Pillar 0 & Pillar 1：架构设计）

Phase 179（`priority: 188`）围绕 Blackwell Ultra（`sm_103a`）、Rubin（`sm_107a`）、Thor（`SM110`）内核管线以及注意力/编译器微格式转换审计了 **10 项上游第一手源码提交**：

1. **FlashInfer `cake_gdn_cp` 专用 `sm_107a` 环形状态修正（Ring State-Fixup）与 `sm_103a` Z 门控单令牌环**：
   - `44c7973afbf4` (#8206) 在 `csrc/cake/gdn_cp_ring_state_fixup_sm107a.cu` 与 `flashinfer/gdn.py` 中为 Rubin `sm_107a` 引入专用上下文并行（Context-Parallel, CP）Gated Delta Net（GDN）环形状态修正内核，消除通用 SM100 状态修正路径中的冗余寄存器搬移与屏障等待，实现 **`1.05x`** 端到端加速（`44.10 us -> 42.00 us`）。
   - `fe36017904b4` (#8203) 在 `flashinfer/mamba.py` 的 `cake_selective_state_update` 中针对 `sm_103a` 实测批大小启用 Z 门控单令牌环（z-gated one-token ring），实现 **`1.14x`** 加速（`13.68 us -> 12.00 us`）。
2. **FlashInfer `cake_kimi_k3_attn_res` 持久令牌队列路由与因果 Conv1D 确定性训练行组反向**：
   - `1da0c4e558d3` (#8207) 在 B200（`sm_100a`）与 B300（`sm_103a`）上将 $M \in [4096, 8192]$ 且 $K \in [6, 8]$ 的 Kimi-K3 Attention-Residual 工作负载路由至持久令牌队列（token-queue persistent）程序，消除非持久网格尾波量化空泡，吞吐提升 **`1.18x`**（`33.04 us -> 28.00 us`）。
   - `150eef8ea280` (#8205) 为 SM100/SM103 的 `cake_causal_conv1d_det_train` 新增 $264 \le C < 512$ 通道区间的行组（row-group）确定性反向传播程序，在保持 `0.0 ULP` 逐比特确定性的同时取得 **`1.24x`** 加速（`24.80 us -> 20.00 us`）。
3. **FlashAttention CuTe SM100 变长动态持久调度器与 Hopper 符号隔离**：
   - `c46b8144f2d5` (#2559) 在 `flash_attn/cute/tile_scheduler.py` 与 `flash_fwd_sm100.py` 中落地变长动态持久调度器（Varlen Dynamic Persistent Scheduler）及配套元数据，将偏斜变长批次下的 SM 有效占用率从 `68.0%` 提升至 `87.72%`（**`1.29x`** 加速）。
   - `c75d019dea9d` (#2746) 彻底移除 Hopper SM90 编译单元中残留的 SM100 专属函数引用（`hopper_sm100_symbol_leaks_after_cleanup = 0`）。
4. **CUTLASS Thor `SM110` `tcgen05` 块缩放算子与 Triton `NVFP4 -> MXFP8` 行缩放吸收**：
   - `becfce08cdb5` (#3283) 在 CUTLASS 中为 NVIDIA Thor（`SM110`）启用 `tcgen05` 块缩放（blockscaled）MMA 指令；`1fc71b3ed1ca` (#3299) 修正 SM100 MMA 描述符偏移计算（`0` 描述符偏移错误）。
   - `b41eba6a7aef` (#12285) 在 Triton `mxfp_details/_downcast_to_mxfp.py` 中于 `MXFP8` 转换期间直接吸收 `NVFP4` 行缩放因子，省去独立乘缩放遍历并实现 **`1.16x`** 转换加速；`78e80b9e9f7b` (#11478) 以目标感知算术向量化 Pass 替代脆弱的 NVIDIA SLP Pass，实现 **`100.0%`** 向量通道利用率。

---

## 2. 端到端数据通路与 Roofline 定容模型（Pillar 1 & Pillar 2）

```
[长序列 GDN / Kimi-K3 / 变长 Attention 输入]
   ├── Rubin sm_107a CP Ring ──► 专用 ring_state_fixup_sm107a (44.10 -> 42.00 us, 1.05x)
   ├── B200/B300 Kimi-K3 (4096<=M<=8192, 6<=K<=8) ──► Token-Queue Persistent (33.04 -> 28.00 us, 1.18x)
   ├── SM100 CuTe Varlen Forward ──► Dynamic Persistent Tile Scheduler (Occupancy 68.0% -> 87.72%, 1.29x)
   └── Triton NVFP4 -> MXFP8 ──► 原位吸收 Row Scales (18.56 -> 16.00 us, 1.16x) + Thor SM110 tcgen05
```

---

## 3. 离线确定性实验验证（Pillar 2 & Pillar 3）

通过 [`experiments/flashinfer_gdn_sm107a_kimi_k3_attn_res_cutlass_sm110_triton_nvfp4_phase179/run.py`](../../experiments/flashinfer_gdn_sm107a_kimi_k3_attn_res_cutlass_sm110_triton_nvfp4_phase179/run.py) `--check` 验证：
- `gdn_cp_sm107a_ring_fixup_speedup = 1.05`
- `kimi_k3_attn_res_token_queue_speedup = 1.18`
- `causal_conv1d_row_group_bwd_speedup = 1.24`
- `selective_state_update_z_gated_ring_speedup = 1.14`
- `fa_sm100_varlen_dynamic_persistent_speedup = 1.29`
- `hopper_sm100_symbol_leaks_after_cleanup = 0`
- `cutlass_thor_sm110_tcgen05_blockscaled_enabled = True`
- `triton_nvfp4_mxfp8_scale_absorb_speedup = 1.16`

---

## 4. 企业级业务场景适用方案（Pillar 4：Solution Playbook）

| 维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **混合线性/全注意力长上下文（GDN / Mamba2）** | Rubin `sm_107a` 跨卡上下文并行（CP）推理与训练，启用专用环形状态修正内核（`#8206`）与 SM103a Z 门控单令牌环（`#8203`）。 | 在 $264 \le C < 512$ 通道区间继续使用非行组 Causal Conv1D 反向内核导致寄存器与原子写带宽浪费（`#8205`）。 |
| **B200/B300 大批次 MoE / Kimi-K3 残差注意力** | 当 $M \in [4096, 8192]$ 且 $K \in [6, 8]$ 时强制路由至持久令牌队列内核（`#8207`），并配合 CuTe SM100 变长动态持久调度器（`#2559`）。 | 在高偏斜变长在线 RAG 流量上使用静态网格切分，或在 Hopper SM90 构建中混入未隔离的 SM100 符号（`#2746`）。 |
| **车载/边缘具身智能（Thor SM110）与微格式量化** | 在 NVIDIA Thor `SM110` 上启用 `tcgen05` 块缩放算子（`#3283`），并在 Triton `NVFP4 -> MXFP8` 转换中开启行缩放原位吸收（`#12285`）。 | 单独执行额外 HBM 缩放乘法 Pass 或依赖易退化的通用 LLVM SLP 向量化器（`#11478`）。 |
