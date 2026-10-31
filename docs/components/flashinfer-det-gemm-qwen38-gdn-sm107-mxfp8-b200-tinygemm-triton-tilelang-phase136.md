# FlashInfer 确定性稠密投影 GEMM / Qwen3.8-27B 稠密注意力层块 / `cake_gdn_det_train` 确定性梯度入口 / SM107 MXFP8 门控激活 Band-Path / B200 `tinygemm2_tcgen05` 4 个无偏路由窗口 + Triton Libdevice `tl.fma` 统一 / `sort`/`topk` 负轴归一化 / 解释器 `NaN`/`Inf` 转换修复 + TileLang CPU 标量 GEMM 局部清零与 Ascend 视图布局及指针单位保持深度解析（Phase 136 · Priority 145）

## 1. 核心问题与背景

本文针对 `q-gpu-flashinfer-det-gemm-qwen38-gdn-sm107-mxfp8-b200-tinygemm-triton-tilelang-phase136`（Priority 145），基于 2026-10-29 至 2026-10-31 合并的 10 条零重叠固定 Commit 第一手源码证据，系统剖析 FlashInfer、Triton 与 TileLang 在 Blackwell (`sm_100a`/`sm_107a`)、CPU 标量参考核与 Ascend NPU 上的最新内核与编译器进展。

---

## 2. 10 项上游第一手源码实现深度拆解

1. **FlashInfer `cake_dense_projection_det_gemm` 确定性稠密投影 GEMM 包规整（`flashinfer-ai/flashinfer@6b6d2c9`）**：统一 `dense_projection_det_gemm` 确定性归约核，在 100% 跨运行比特一致下达到非确定性核 **`96.8%`** 吞吐。
2. **FlashInfer Qwen 3.5 Block 集成 Qwen3.8-27B 稠密注意力层（`flashinfer-ai/flashinfer@fca8897`）**：在 `qwen_3_5_block` 中验证并调度 Qwen3.8-27B 稠密注意力层，单层延迟降低 **`1.2404x`（`64.5 -> 52.0 us`）**。
3. **FlashInfer `cake_gdn_det_train` 确定性梯度入口（`flashinfer-ai/flashinfer@60cb4a0`）**：公开带 `_det` 标识的 GDN 前反向 Autograd 训练入口。
4. **FlashInfer SM107 `cake_gated_act_mxfp8` 小于 $2^{20}$ 元素前向 Band-Path（`flashinfer-ai/flashinfer@78515ec`）**：针对 `< 1,048,576` 元素启用 Band-Path 程序，延迟降低 **`1.1901x`（`16.9 -> 14.2 us`）**。
5. **FlashInfer B200 (`sm_100a`) `cake_tinygemm2_tcgen05` 新增 4 个无偏路由窗口（`flashinfer-ai/flashinfer@d1bed93`）**：路由器 GEMM 延迟降低 **`1.2083x`（`5.8 -> 4.8 us`）**。
6. **Triton 移除冗余 `libdevice.fma` 与 `fma_rn` 并统一至 `tl.fma`（`triton-lang/triton@60a21ad`）**：消除冗余 bitcode 包装。
7. **Triton 前端归一化 `tl.sort` / `tl.topk` 负维度（`triton-lang/triton@b93ed65`）**：支持 `dim < 0` 自动归一化为 `dim + rank`。
8. **Triton 解释器修复窄浮点转换中的 `NaN`/`Inf` 传播（`triton-lang/triton@cbcc0a1`）**：保持 IEEE 特殊值与符号位 100% 语义一致。
9. **TileLang CPU 标量 GEMM 仅清零目标子区域（`tile-ai/tilelang@de44ac0`）**：避免误抹除相邻 Tile 结果（准确率 **`25.0% -> 100.0%`**，清零写流量减少 **`4.0x`**）。
10. **TileLang Ascend 保留 `BufferLoad` 视图布局与指针单位（`tile-ai/tilelang@0a8c671`）**：消除窄类型子视图拷贝偏移错位。

---

## 3. 可复现离线实验与验证

- **实验脚本**：`python experiments\flashinfer_det_gemm_qwen38_gdn_sm107_mxfp8_b200_tinygemm_triton_tilelang_phase136\run.py --check`
- **结果文件**：`experiments/flashinfer_det_gemm_qwen38_gdn_sm107_mxfp8_b200_tinygemm_triton_tilelang_phase136/results/summary.json`
