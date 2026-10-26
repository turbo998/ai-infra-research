# FlashInfer Cake QSA NVFP4、MiniMax-H3 SM103 4-Dest Norm、Grouped MXFP8 提前退出、FP8 Attention 折叠 K-Mean Bias、GDP 尾波段与 TileLang 双射/分析器防护（Phase 96）

> **证据分类**：`implementation`（固定上游源码提交）+ `own-offline`（本项目纯 Python 3.12 标准库离线确定性验证，非 GPU 硬件实测 profiling）。

## 1. 核心上游源码实证（10 项零重叠一手提交）

| Source ID | 仓库 / 提交 | 关键实现路径与架构机制 |
|---|---|---|
| `flashinfer-cake-qsa-nvfp4-early-blocktable-partial-epilogue-pr7460-2026` | `flashinfer-ai/flashinfer` `e400000d1c` (PR #7460) | 在 SM100/SM103 (`compute capability 10.0 / 10.3`) 的 `cake_qsa_nvfp4` `tcgen05` decode kernel 中，将 paged KV 的 block-table 索引加载提前至 TMA prologue 之前，并在 epilogue 中直接写出 split-KV partial 结果，省去 1 次 shared-memory 中转往返（`direct_epilogue_smem_roundtrips = 0`）。 |
| `flashinfer-cake-minimax-h3-sm103-4dest-nvfp4-norm-pr7458-2026` | `flashinfer-ai/flashinfer` `2751459afe` (PR #7458) | 在 `cake_minimax_h3` pre-attention 阶段，当输出目标分区数 `P == 4` 且运行于 `sm_103a` (B300) 时，路由至专属 4-destination NVFP4 `NORM` kernel（配合 `FUSED_PACK_K96`/`FUSED_E4M3_K96` GEMM）。 |
| `flashinfer-cake-grouped-mxfp8-early-exit-prefetch-partial-pr7457-2026` | `flashinfer-ai/flashinfer` `32f46c223f` (PR #7457) | `cake_grouped_mxfp8_quantize` 的 per-K row-tiled 程序在 CTA 起始行超出动态有效行数（`row_start >= valid_rows`）时立即退出，并对跨 expert 边界的 partial group 执行预取。 |
| `flashinfer-cake-fp8-attention-fold-kmean-bias-exp2-pr7429-2026` | `flashinfer-ai/flashinfer` `050e27087d` (PR #7429) | `cake_fp8_attention` 将 smoothed-K 均值偏置 $q \cdot \mu_K$ 直接折叠进 `exp2` 的标度偏置参数（$2^{(q\cdot k - q\cdot \mu_K)\log_2 e \cdot s}$），并对 `stats_finalize` 执行 `unroll = 8` 展开。 |
| `flashinfer-cake-trtllm-mla-blackwell-aligned-dense-decode-pr7427-2026` | `flashinfer-ai/flashinfer` `1769ac7474` (PR #7427) | `cake_trtllm_mla_blackwell` 针对 `sm_100a` 与 `sm_103a` 特化对齐稠密 Multi-head Latent Attention (`MLA`) decode 调度程序。 |
| `flashinfer-cake-gdp-tail-wave-band-gate-logit-carrier-pr7393-2026` | `flashinfer-ai/flashinfer` `69e33dae7b` (PR #7393) | `cake_gdp`（Gated Delta Product，$K=128, V=64, 1\le n\le 4$ Householder 步）制定 `T = 1` 尾波段规则：当 $4 \cdot \text{SM} < B \cdot HV \le 5 \cdot \text{SM}$ 时路由至每线程单行 `l8r4p` 程序，并为 `T = 4` 引入 gate-logit carrier、为 SM100 `T = 8` 引入 dense-load carrier。 |
| `flashinfer-cake-gated-act-mxfp8-packed-f32x2-sigmoid-pr7396-2026` | `flashinfer-ai/flashinfer` `93e5508a15` (PR #7396) | `cake_gated_act_mxfp8` 在 `sm_100a`（4 个）、`sm_103a`（9 个）、`sm_107a`（5 个）共 18 个前向/反向 SwiGLU+MXFP8（`128x4` swizzled E8M0 `RCEIL` 缩放）特化程序中启用打包 `f32x2` sigmoid 计算。 |
| `tilelang-revert-nan-propagating-clamp-lowering-pr3628-2026` | `tile-ai/tilelang` `a2cf4c2b1e` (PR #3628) | 将 `T.clamp(x, lo, hi)` 恢复为标准 `T.min(T.max(x, lo), hi)` lowering，消除跨后端 codegen 回归。 |
| `tilelang-reject-non-bijective-sm8x-sparse-metadata-pr3125-2026` | `tile-ai/tilelang` `ced0ca1a30` (PR #3125) | 在 `make_cutlass_metadata_layout_sm8x` 中强制校验 `M % group == 0`（16-bit 下 `group = 32`，8-bit 下 `group = 16`）且 `K % 2 == 0`，防止非双射 2:4 稀疏 metadata 交错布局静默越界。 |
| `tilelang-avoid-stale-analyzer-bindings-loop-partition-pr3226-2026` | `tile-ai/tilelang` `741bf8b722` (PR #3226) | `LowerParallelLoop` 与 `GetVectorizeSize` 改为接收 `const arith::Analyzer *analyzer` 并构造局部快照，防止并行循环分区改写后残留陈旧的循环变量边界绑定。 |

## 2. 数学建模与架构师计算练习

### 2.1 FP8 Attention K-Mean Bias 折叠与 Grouped MXFP8 提前退出收益
1. **指数参数折叠恒等式**：
   $$2^{(q \cdot k)\log_2 e \cdot s} \cdot 2^{-(q \cdot \mu_K)\log_2 e \cdot s} = 2^{((q \cdot k - q \cdot \mu_K)\log_2 e \cdot s)}$$
   在采样分值区间 $[-1.25, 2.4]$ 上，折叠前后最大绝对误差 $< 10^{-14}$，同时消除了 softmax 主循环外的独立乘法修正。
2. **Per-K Row-Tiled MXFP8 提前退出**：
   当 MoE 路由缓冲区按最大上限 `max_padded_rows = 1024` 分配、每 CTA 处理 `tile_m = 128` 行（每 K 分块启动 $8$ 个 CTA），而实际动态有效 token 数 `valid_rows = 384` 时，仅前 $\lceil 384 / 128 \rceil = 3$ 个 CTA 执行量化，其余 $5$ 个 CTA 在序言阶段立即退出，节省 $5 / 8 = 62.5\%$ 的无效 CTA 周期。

### 2.2 Cake GDP 尾波段（Tail-Wave Band）调度准则
在 148-SM Blackwell (`sm_100a`) 上，`l8r4w5` 配置支持每 SM 驻留 5 个 CTA，而 `l8r4p`（每线程单行）配置每 SM 驻留 4 个 CTA 但单 CTA 延迟更低。当解码网格满足：
$$4 \times 148 = 592 < B \cdot HV \le 5 \times 148 = 740$$
例如 $B \cdot HV = 656$ 时，选择 `one_row_per_thread_l8r4` 可避免 5-CTA/SM 模式下寄存器与共享内存争用导致的尾波放大。

## 3. 可复现离线实验与验证

运行命令：
```bash
python experiments/flashinfer_qsa_nvfp4_minimax_mxfp8_fp8_attn_mla_gdp_gated_act_tilelang_phase96/run.py --check
```
实验产物 `summary.json` 已验证：
- `direct_epilogue_smem_roundtrips == 0`，`qsa_decode_speedup_ratio == 1.1556`；
- `saved_cta_fraction == 0.625`，`total_packed_programs == 18`（`sm_100a: 4`, `sm_103a: 9`, `sm_107a: 5`）；
- `selected_gdp_t1_program == "one_row_per_thread_l8r4"`（当 $B \cdot HV = 656 \in [593, 740]$）；
- `sm8x_valid_fp16_64x128 == True`，`sm8x_invalid_fp16_48x128 == False`，`sm8x_invalid_fp16_64x127 == False`。
