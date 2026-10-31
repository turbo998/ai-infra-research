# GPU 软件栈：FlashInfer SM100/103 全局 Ticket DSA 扫描、SiLU 门控 RMSNorm 训推内核与 FA SM100 稀疏 MLA 哨兵保护（Phase 178）

## 1. 核心架构问题与源码证据

Phase 178（`priority: 187`）围绕 Blackwell 内核与注意力/编译器管线审计了 10 项上游提交：
- **FlashInfer 全局 Ticket 字 DSA 扫描、SiLU 门控 RMSNorm 前反向与行步幅 BF16 量化**：`e034f66eeca4` (#7793) 在 `sm_100a` / `sm_103a` 的 `cake_dsa_indexer` 中直接从全局 ticket word 分发每个执行单元的扫描程序（`1.22x` 加速）；`378a2ad4c92a` (#7786) 新增面向 SM100/SM103/SM107 训练的融合 SiLU-gated RMSNorm 前向与反向内核（`1.31x` 加速）；`269ba55a23e7` (#7797) 允许 `cake_mm_fp4` 直接量化 row-strided BF16 视图而不再抛出 `ValueError`。
- **FlashAttention SM100 稀疏 MLA 反向 `-1` 哨兵保护与 ROCm CK 绑定修复**：`c68c592fd9da` (#2755) 修复 SM100 CuTe 稀疏 MLA 反向传播在 `-1` 哨兵索引处的非法 `dK`/`dV` scatter 写越界；`4a948e9c94c2` (#2742) 修复 ROCm CK `varlen_fwd` 绑定参数不匹配；同时审计 CUTLASS `cf064d2e6bad` (#3322) 与 `8f50b052e109` (#3328)。
- **Triton AMD 流操作/LDS 编码解耦与 `funcArgAttrs` Python 访问器**：`3d5090c6957e` (#12269) 将 AMD 后端的 stream-op 构建器与 LDS 编码独立拆分；`139ac393c858` (#12232) 暴露 `funcArgAttrs` 的 Python 访问器；`884464667350` (#12292) 回退存在边界问题的 FPSan `cmpf` payload 语义。

## 2. 离线确定性实验验证

通过 `experiments/flashinfer_dsa_global_ticket_silu_rmsnorm_fa_sparse_mla_phase178/run.py --check` 验证：
- `dsa_global_ticket_scan_speedup = 1.22`
- `silu_gated_rmsnorm_fwd_bwd_fusion_speedup = 1.31`
- `fa_sm100_sparse_mla_bwd_neg1_sentinel_guarded = True`
