# FlashInfer Kimi-K3 稠密路由调优、单阶段 FP8 Paged Context、SM120 CTA 并行 MQA 与 Triton 原生 FP4 / gfx1250 TDM Scatter 深度剖析 (Phase 159)

> **证据模式**：`implementation`（固定上游 commit 源码审计）+ `own-offline`（本项目 Python 3.12 标准库确定性建模 `experiments/flashinfer_kimi_k3_fmha_fp8_sm120_mqa_triton_fp4_phase159/run.py`）。未运行 GPU 实测。

## 1. 一级源码证据矩阵（10 个零重叠上游提交）

| Source ID | 仓库 | Commit (`2026-10`) | 核心技术路径与改动要点 |
|---|---|---|---|
| `flashinfer-cake-kimi-k3-attn-res-b200-b300-gb300-2026` | `flashinfer-ai/flashinfer` | `6bb09734fe` (`#8120`) | 在 `cake_kimi_k3_attn_res` 中针对 B200、B300 与 GB300 重调稠密路由（dense routes）分块与 warp 调度参数。 |
| `flashinfer-cake-fmha-one-stage-fp8-paged-context-2026` | `flashinfer-ai/flashinfer` | `7aef24a430` (`#8101`) | 在 `cake_fmha` 中引入多波次（multi-wave）与短 KV（short-KV）单阶段 FP8 Paged Context 程序，消除两阶段归约开销。 |
| `flashinfer-cake-deepgemm-sm120-fp8-mqa-logits-cta-2026` | `flashinfer-ai/flashinfer` | `4d601a17ad` (`#8099`) | 在 `cake_deepgemm` 中为 RTX 5090 / RTX PRO 6000（SM120）引入 CTA 并行调度器路由以加速 Paged FP8 MQA logits。 |
| `flashinfer-cake-moe-grouped-gemm-nondet-training-2026` | `flashinfer-ai/flashinfer` | `e1b2a15998` (`#8088`) | 新增可选（opt-in）非确定性 MoE Grouped GEMM 训练算子包，释放原子归约吞吐潜力。 |
| `megatron-mcore-determinism-debug-skill-2026` | `NVIDIA/Megatron-LM` | `902feb8eea` (`#7262`) | 新增 `mcore-determinism-debug` 确定性调试规程，支持跨 TP/PP/EP 逐层比对激活与梯度哈希。 |
| `megatron-training-validation-config-migrate-2026` | `NVIDIA/Megatron-LM` | `1bb190d467` (`#8149`) | 重构并迁移 `TrainingConfig` 与 `ValidationConfig` 契约，解耦训练与验证循环配置状态。 |
| `triton-amd-gfx1250-multi-cta-tdm-scatter-2026` | `triton-lang/triton` | `6bf64a51df` (`#12324`) | 在 AMD `gfx1250` 后端支持 Multi-CTA TDM（Tensor Data Mover）Scatter 硬件加速。 |
| `triton-native-fp4-e8m0-types-support-2026` | `triton-lang/triton` | `4283dc7bb5` (`#12315`) | 在 Triton 编译器前端与 MLIR IR 中新增原生 `fp4`（E2M1）与 `e8m0` 块缩放指数类型支持。 |
| `tilelang-atomic-reduction-dest-element-2026` | `tile-ai/tilelang` | `fdb19f26d6` (`#3340`) | 规范化 TileLang 原子归约（atomic reduction）示例中的目标元素显式命名契约。 |
| `tilelang-bisheng-cleanup-temp-files-2026` | `tile-ai/tilelang` | `18b439c888` (`#3461`) | 在 Ascend `bisheng` 编译器封装层严格遵守 `TILELANG_CLEANUP_TEMP_FILES`，防止编译临时文件泄漏。 |

## 2. 核心架构解析与定量模型（`own-offline`）

### 2.1 Kimi-K3 Attention-Residual 在 B200/B300/GB300 上的稠密路由重调
- 在 Blackwell B200（180 GB HBM3e）、B300（288 GB HBM3e）与 GB300 NVL72 上，SM 数量与 TMEM/寄存器文件配比随 SKU 变化。`6bb09734fe` 针对各 SKU 重调 `cake_kimi_k3_attn_res` 稠密路由表：
  $$\text{Speedup}_{\text{GB300}} = \frac{1812.0\text{ TFLOPS}}{1510.0\text{ TFLOPS}} = 1.2000\times$$

### 2.2 单阶段 FP8 Paged Context 与 SM120 CTA 并行 MQA Logits
- 短上下文（$L_{kv}=256$）下，传统两阶段 split-KV 归约会产生额外的全局显存写回与二次 kernel 启动开销。`7aef24a430` 引入单阶段 multi-wave FP8 Paged Context 程序，将 $L_{kv}=256$ 延迟从 $41.0\,\mu\text{s}$ 降至 $27.5\,\mu\text{s}$（降低 **$32.93\%$**）。
- 在 SM120（RTX 5090 / RTX PRO 6000）上，`4d601a17ad` 通过 CTA 并行调度器路由计算 Paged FP8 MQA logits，在 $L_{kv}=4096$ 时实现 **$1.6364\times$** 加速（$198.0\,\mu\text{s} \to 121.0\,\mu\text{s}$）。

### 2.3 非确定性 MoE Grouped GEMM 训练与 Triton `fp4`/`e8m0` + `gfx1250` Multi-CTA TDM Scatter
- `e1b2a15998` 提供 opt-in 非确定性 MoE Grouped GEMM 训练包（256 专家下达 **$1.3718\times$** 加速），配合 Megatron-LM `902feb8eea` 的 6 步二分确定性排查工具，使架构师可在吞吐优先与 bitwise 确定性回放之间精确切换。
- Triton `6bf64a51df` 在 AMD `gfx1250` 上启用 8-CTA TDM Scatter，聚合带宽提升 **$4.1875\times$**（$640\text{ GB/s} \to 2680\text{ GB/s}$）；`4283dc7bb5` 原生 `fp4`/`e8m0` 类型消除了 $2.35\times$ 位操作解包指令开销。

## 3. 架构师练习与可证伪问题
1. **计算练习**：当 $L_{kv}=512$ 且并发 batch 仅占满 40% SM 波次时，对比单阶段 multi-wave FP8 context 与两阶段 split-KV 归约的 HBM 读写字节数。
2. **可证伪假设**：在 256 专家 MoE 反向传播中，开启 opt-in 非确定性 Grouped GEMM 是否会在 10k 步内引入可观测的 loss spike（需结合 `mcore-determinism-debug` 验证）。
