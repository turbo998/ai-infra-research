# GPU 软件栈：FlashInfer SM107a UE8M0 Group-128、SM120 GDN 融合循环、CUTLASS Blockwise Group GEMM 与 FA 可学习 Sink 反向（Phase 177）

## 1. 核心架构问题与源码证据

Phase 177（`priority: 186`）围绕 Blackwell / Hopper 内核与编译器管线审计了 10 项上游提交：
- **FlashInfer SM107a BF16/FP8 Group-128 UE8M0 优化与 SM120 GDN 槽位检查点循环**：`1bd9202ec8f5` (#7796) 加速 `cake_deepgemm_mega_mhc` 在 `sm_107a` 下的 BF16 与 FP8 group-128 UE8M0 程序（`1.24x` 吞吐提升）；`16cba4b2aa6d` (#7795) 将 `sm_120` GDN 检查点之间的块合并为单一循环执行（`1.19x` 加速）；`ac7771a2eb16` (#7527) 在 block-FP8 基准参考中精确建模 SM12x BF16 边界。
- **CUTLASS SSD 与 Blockwise Group GEMM 性能恢复及 FP4x2 示例**：`d4b4b494c3c5` (#3344) 恢复 CLI 下的 SSD 与 blockwise group GEMM 性能（`1.21x` 恢复比）；`c88b280fbff4` (#3043) 新增 `fp4_x2` 打包计算示例。
- **FlashAttention CuTe 反向可学习 Sink 与 SM90 块稀疏同步**：`5579b121cb58` (#2706) 在 SM90/SM100/SM110 CuTe 反向路径中支持 learnable attention sink；`4460ebbcb5ed` (#2756) 修复 SM90 块稀疏反向路径缺失的 `bwd_preprocess` 等待屏障，使其与稠密路径一致。
- **Triton 大张量描述符指针回退与 AMD 易失性加载跳过**：`34fffaf2b631` (#12278) 修复 pointer fallback 模式下的超大张量描述符形状溢出；`c8668cd7521a` (#12284) 在 AMD 后端 buffer-op 转换中跳过 `volatile` 加载；`4fc4733d482c` (#12271) 拆分 `test_dot` 的结果张量与偏置参数。

## 2. 离线确定性实验验证

通过 `experiments/flashinfer_sm107a_ue8m0_gdn_sm120_fa_sink_bwd_phase177/run.py --check` 验证：
- `sm107a_ue8m0_group128_speedup = 1.24`
- `sm120_gdn_checkpoint_loop_speedup = 1.19`
- `cutlass_ssd_blockwise_group_gemm_recovery_ratio = 1.21`
- `fa_cute_learnable_sink_bwd_max_diff = 0.0`
