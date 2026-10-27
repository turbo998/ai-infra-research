# FlashInfer 融合 Norm-Combine、MiniMax-H3 SM120 预注意力 GEMM、NVFP4 L2 预取、Kimi-K3 TP12 持久归约与 MLA、MoE AR Union、CUTLASS 及 TileLang（Phase 110）

> English version: [flashinfer-norm-combine-minimax-fp4-kimi-mla-cutlass-sm120-tilelang-phase110.en.md](flashinfer-norm-combine-minimax-fp4-kimi-mla-cutlass-sm120-tilelang-phase110.en.md) | 关联实验：`experiments/flashinfer_norm_combine_minimax_fp4_kimi_mla_phase110/run.py`

## 1. 一手代码证据（2026-08-06 至 2026-10-27）

| 仓库 | Commit | 日期 | 核心机制变更 | 证据等级 |
|---|---|---|---|---|
| `flashinfer` | `3da22ba71f` | 2026-10-27 | `cake_fused_norm_combine` 精简 38..479 token stream 程序序言（SMEM `128 -> 256 B`，含 `8 B` handoff）并在 1-token 程序提前读取 workspace 表（B200 加速 `1.026x-1.034x`） | `implementation` + `author-benchmark` |
| `flashinfer` | `5b2aacd211` | 2026-10-27 | `cake_minimax_h3` 优化 `sm_120a` FP8/NVFP4 预注意力 GEMM epilogue（更新 6/9 个程序，RTX PRO 6000 加速最高 `1.0798x`，RTX 5090 最高 `1.0469x`） | `implementation` + `author-benchmark` |
| `flashinfer` | `e158799310` | 2026-10-27 | `cake_mm_fp4` 在 NVFP4 per-token 量化+GEMM 链新增 3 个 L2 权重入口预取单元（`sm_100a` `n8_aF_l2256b_o2` 扩至 `M=1..8`，`sm_103a` `m64_bF_l2256b_ro` 覆盖 `M=128`，独立 GEMM 新增 8 行 `M=128` 规则） | `implementation` + `author-benchmark` |
| `flashinfer` | `531e0bc2ba` | 2026-10-27 | `cake_kimi_k3_tp12_tail` 为 `320 <= M <= 960` 引入无 scatter 阶段的持久归约尾核 `k3_reduce:grouped`（双 token gather 在途，GB300 `M=960` 加速 `1.056x`） | `implementation` + `author-benchmark` |
| `flashinfer` | `45411ad7d7` | 2026-10-27 | `cake_kimi_k3_mla` 为 `16/32/48` 行 tile 引入延迟触发（`main CTAs >= 32`）与等待前预计算行偏移的 split-KV 归约器（Shape 1 在 B200 加速 `1.1649x`、GB300 加速 `1.1295x`） | `implementation` + `author-benchmark` |
| `flashinfer` | `acb2bef7ec` | 2026-10-27 | `cake_trtllm_moe_allreduce_union` 单 token 行采用 `2 CTAs/SM`（`LAUNCH_MIN_BLOCKS=2`）与 `st.async` RMS 交换，全部 30 个测试形状在 B200/B300 加速 `1.03x-1.13x` | `implementation` + `author-benchmark` |
| `cutlass` | `976d0521f0` | 2026-08-06 | `cute::operator==(ScaledBasis, ScaledBasis)` 仅在基轴相等时比较值类型，并在 `sm120_blockscaled_sparse_mma_tma.hpp` 补充 `TileShape_K` 整除元数据原子 `K`（`256` 或 `128`）静态断言 | `implementation` |
| `cutlass` | `3c063071bd` | 2026-08-06 | 修复 `AffineRank2RowMajor::packed(extent)` 步长顺序为 `(extent.column(), 1)`，消除误用列主序 `(1, extent.row())` 导致的线性偏移错位 | `implementation` |
| `tilelang` | `cb6455d680` | 2026-10-19 | 修复 warp specialization 下共享内存 `T.cumsum` 扫描谓词：传入 `thread_offset = thread_bounds->min` 并判断 `threadIdx.x - thread_offset >= SEG` | `implementation` |
| `tilelang` | `9253431f2a` | 2026-10-19 | ROCm HIP 代码生成支持 `float4_e2m1fn` 常量打印（发射 `__tl_float_to_fp4(...)` 并在 `VisitExpr_(FloatImmNode)` 置位 `enable_fp4_` 自动引入 `hip_fp4.h`） | `implementation` |

## 2. 关键实现与架构师结论

1. **PDL 链式预取与 Split-KV 归约延迟触发**：`cake_mm_fp4` 在 PDL 等待前将首个 tile 的前 `4` 个 K-stage 权重预取入 L2；`cake_kimi_k3_mla` 在主网格 `>= 32 CTAs` 时延迟触发 split-KV 归约器并提前完成归约器行地址计算，将单 batch 12 头 MLA 解码在 B200 上提速 `1.1649x`。
2. **Warp Specialization 线程偏移与 Layout 步长不变量**：在 Hopper/Blackwell 消费者 warp（如 `thread_offset = 128`）上运行 `InclusiveScan1D/2D` 时，若直接判断 `threadIdx.x >= 32` 会导致 `0` 个活跃线程执行扫描；减去 `thread_offset` 后完整恢复 `32` 个扫描 lane。CUTLASS `AffineRank2RowMajor::packed(64, 128)` 在坐标 `(3, 5)` 处将偏移从错误的 `323` 修正为 `389`。
