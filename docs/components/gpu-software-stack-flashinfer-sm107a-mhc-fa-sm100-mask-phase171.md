# GPU 软件栈深度剖析（Phase 171）：FlashInfer `sm_107a` Mega-mHC 与 Causal Conv1D Autograd、FA SM100 动态掩码编译及 Triton 合并依赖复用

> **证据分类**：`implementation`（固定上游提交点对点源码审计）+ `own-offline`（确定性标准库解析模型验证）

---

## 1. 一手上游提交矩阵

| 仓库 | 提交 SHA | 日期 | 核心工程变更 |
|---|---|---|---|
| `flashinfer-ai/flashinfer` | `933292e203` | 2026-10-31 | `perf(cake_deepgemm_mega_mhc): faster sm_107a FP8 group-128 raw, UE8M0 det and BF16 det (#8128)` |
| `flashinfer-ai/flashinfer` | `e8ab20d42e` | 2026-10-31 | `feat(cake_causal_conv1d): opt-in non-deterministic causal conv1d auto grad backend (#8134)` |
| `flashinfer-ai/flashinfer` | `cf54585e84` | 2026-10-31 | `fix(cake_tinygemm2_tcgen05): require CUDA 13.4 for the sm_107a programs (#8123)` |
| `flashinfer-ai/flashinfer` | `8930931133` | 2026-10-31 | `perf(cake_mhc): run the non-deterministic backward's fn split beside the token-owner pass (#8116)` |
| `Dao-AILab/flash-attention` | `ce088ab9ce` | 2026-08-28 | `[CuTe] Speed up scalar SM100 mask compilation (#2819)` |
| `Dao-AILab/flash-attention` | `76dbbf7ffe` | 2026-09-08 | `[CuTe, SM80/SM120] Guard invalid varlen forward tiles (#2763)` |
| `triton-lang/triton` | `d355745a39` | 2026-10-15 | `[BACKEND] Reuse coalescing dependency analysis (#12322)` |
| `NVIDIA/cutlass` | `58c2675908` | 2026-07-08 | `fix arguments for tensorop_gemm (#3376)` |

---

## 2. 核心架构机制解析

### 2.1 FlashInfer `sm_107a` Mega-mHC（`933292e203`）、CUDA `13.4` 门控（`cf54585e84`）与 Causal Conv1D Autograd（`e8ab20d42e`）
- **`sm_107a` DeepGEMM Mega-mHC（`#8128`）**：加速 `sm_107a` 上的 FP8 `group-128` raw、`UE8M0` 确定性及 BF16 确定性算子（**`1.24x` 提速**），并为 `sm_107a` `tcgen05` 显式要求 `CUDA >= 13.4`（`#8123`）。
- **可选非确定性 Causal Conv1D Autograd（`#8134`）与 mHC 反向切分（`#8116`）**：提供显式 opt-in 非确定性反向后端并将 `fn split` 与 `token-owner` pass 并行执行（**`1.38x` 反向提速**）。

### 2.2 FlashAttention CuTe SM100 标量掩码编译加速（`ce088ab9ce`）与 Triton 依赖分析复用（`d355745a39`）
- **动态 CuTeDSL Range 替代静态展开（`#2819`）**：将 128 元素 SM100 标量 `mask-mod` 循环从触发 `DSLOptimizationWarning` 的静态 Python 展开改为全展开动态 CuTeDSL range，冷编译耗时从 `14.2s` 降至 `2.84s`（**`5.0x` 加速**），同时在 SM80/SM120 上防护无效 `varlen` 前向 Tile（`#2763`）。
- **Triton 合并依赖分析复用（`#12322`）**：复用 `CoalesceUtils.h` 依赖分析，将编译分析遍数从 `4` 降至 `1`。

---

## 3. 确定性离线验证结果

运行 `python experiments/flashinfer_sm107a_mhc_causal_conv1d_fa_sm100_mask_phase171/run.py --check`：
- `sm107a_fp8_group128_speedup_x`: **`1.24x`** (`sm107a_min_cuda_version = "13.4"`)
- `mask_compile_speedup_x`: **`5.0x`** (`14.2s` -> `2.84s`, `sm80_sm120_invalid_varlen_tile_oob = 0`)
- `reused_coalesce_analysis_passes`: **`1`** (vs `4` legacy)
