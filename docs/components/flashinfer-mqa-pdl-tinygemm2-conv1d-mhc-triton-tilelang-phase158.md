# FlashInfer SM100a MQA 并行调度、PDL QK-RoPE-Append 重叠、SM107a TinyGEMM2 路由、Causal Conv1D `bwd_grid_mult`、mHC 图捕获防护与 Triton/TileLang 编译器原语（Phase 158）

> **English Version**: [flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-phase158.en.md](flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-phase158.en.md)
> **关联问题 ID**：`q-flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-phase158` (`priority: 167`)
> **关联证据 ID**：`ev-flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-2026-phase158`

---

## 1. 架构背景与核心发现

在 Blackwell (`sm_100a` / `sm_103a`) 与 Vera Rubin (`sm_107a` / `R200`) 的低延迟推理解码及多流并发训练中，尾波次（2nd-wave）SM 争用、相邻算子间的流水线排空（Drain Bubble）以及 CUDA Graph 捕获期间的脏状态残留是决定端到端 P99 延迟的关键瓶颈。本报告基于 2026-10-02 至 2026-10-31 合并的 10 个主线提交，系统拆解以下五项底层机制：

1. **SM100a Cake DeepGEMM Dense MQA Partial-Block 并行调度（`flashinfer@9d55883`, `#8113`）**：将 `cake_deepgemm_dense_mqa_1974e6db96c173903467` 并行调度 logits 变体接入 `fp8:h64:partial:qle128:kvsplit` 桶（上限从 $K \le 4096$ 扩展至 $K \le 8192$），在 B200 上实现 **1.0506x–1.0634x** 加速（中位数 **1.0563x**，`4.83 -> 4.54 us`），且 `106/106` 形状逐比特一致。
2. **SM100a/SM103a Fused QK RMSNorm + RoPE + Paged KV Append 的 PDL 重叠发射（`flashinfer@b194ced`, `#8112`）**：引入 `enable_pdl: bool = False` 选通参数，利用 `cudaTriggerProgrammaticLaunchCompletion` 与 `cudaGridDependencySynchronize()` 将上游 cuBLAS QKV 投影 GEMM 尾波与融合 RoPE+KV 追加内核重叠，在 B200/GB200/B300/GB300 上实现 FP8 几何平均 **1.1395x–1.1499x**（峰值 **1.3558x**，`6.768 -> 4.992 us`）与 BF16 几何平均 **1.1720x–1.1886x**（峰值 **1.3358x**，`6.520 -> 4.881 us`）加速。
3. **SM107a (`R200`) TinyGEMM2 `tcgen05` 9–16 行无偏置路由窗口（`flashinfer@35be896`, `#8110`）**：在 `sm_107a` 上新增 `M ∈ [9, 16], N ∈ [4097, 16384], K ∈ [1024, 16384]` 的 `PACKED` 与 `PACKED_M64_N16`（当 $N > 6784$ 且 $K > 12288$）两级分发分支。
4. **Cake Causal Conv1D 确定性反向传播 `bwd_grid_mult` 网格拆分（`flashinfer@d29e16a`, `#8104`）**：针对多流并发（如通信辅助流占用 1–32 个 CTA）导致最后 1–2 个 CTA 滑入第 2 波次的问题，通过 `bwd_grid_mult ∈ {1, 2, 3, 4}` 沿 token 轴进一步切分 `grid.x`，在 R200 上取得 **1.43x**（`0.1589 -> 0.1114 ms`）、GB300 上 **1.28x**（`0.7072 -> 0.5526 ms`）、B200 上 **1.27x**（`0.7813 -> 0.6175 ms`）加速，同时保持 `y` 与 `dx` 逐比特确定性一致。
5. **Cake mHC CUDA Graph 捕获期零分配守卫与 Triton/TileLang 原语对齐（`flashinfer@fc6d231`, `triton@e6aa06c`, `triton@568e9d3`, `tilelang@b95ee4f`, `tilelang@c62ebe9`, `tilelang@256791f`）**：禁止在 `torch.cuda.is_current_stream_capturing()` 期间首次惰性分配 mHC 票据缓冲区，防止捕获流中的 `torch.zeros` 未在默认流回放而遗留脏票据字。

---

## 2. 权威上游证据与定量指标对比

| 仓库与提交 | 关键函数 / 模块路径 | 硬件 / 目标架构 | 核心机制与定量效果 |
| :--- | :--- | :--- | :--- |
| `flashinfer-ai/flashinfer@9d55883` (`#8113`) | `flashinfer/jit/cake_deepgemm_dense_mqa.py` | B200 (`sm_100a`, 148 SM) | `fp8:h64:partial:qle128:kvsplit` 桶启用并行调度（$K \le 8192$），中位数 **1.0563x**（`4.83 -> 4.54 us`），`106/106` bitwise equal |
| `flashinfer-ai/flashinfer@b194ced` (`#8112`) | `csrc/cake_fused_qk_rope_append_kernel.cu` | B200 / GB200 / B300 / GB300 (`sm_100a`/`sm_103a`) | `enable_pdl=True` 启用 PDL，FP8 geomean **1.1395x–1.1499x**（最高 **1.3558x**），BF16 geomean **1.1720x–1.1886x**（最高 **1.3358x**） |
| `flashinfer-ai/flashinfer@35be896` (`#8110`) | `csrc/cake_tinygemm2_tcgen05.cu` | R200 (`sm_107a`) | `M ∈ [9, 16]` 无偏置 GEMM 路由至 `PACKED` 与 `PACKED_M64_N16`（$N > 6784, K > 12288$） |
| `flashinfer-ai/flashinfer@d29e16a` (`#8104`) | `flashinfer/cake/causal_conv1d_det.py` | B200 / GB300 / R200 | `bwd_grid_mult ∈ {1,2,3,4}` 缓解旁路流尾波惩罚：R200 **1.43x**，GB300 **1.28x**，B200 **1.27x** |
| `flashinfer-ai/flashinfer@fc6d231` (`#8100`) | `flashinfer/cake/mhc.py` | CUDA Graph Stream | `_get_default_workspace()` 在捕获未命中时抛出 `ValueError`，消除未初始化票据死锁（`550/550` 测试通过） |
| `triton-lang/triton@e6aa06c` (`#12456`) | `python/triton_kernels/triton_kernels/matmul.py` | Triton JIT Backend | 移除冗余 `enable_fp_fusion` 覆盖，统一复用后端默认融合语义 |
| `triton-lang/triton@568e9d3` (`#12403`) | `python/triton/runtime/jit.py` | Triton AST Frontend | 将 Python `bool` 注入 `builtin_namespace` 并映射为 `tl.int1`（降低为 `arith.select`） |
| `tile-ai/tilelang@b95ee4f` (`#3361`) | `src/target/rt_mod_hip.cc`, `make_packed_api.cc` | TileLang TVM FFI | `dtype.is_bfloat16()` 标量参数通过 `Float(64)` 解包后再强转为 `bfloat16_t` |
| `tile-ai/tilelang@c62ebe9` (`#3517`) | `testing/python/transform/*.py` | TileLang CUDA Test | 源码检查测试改用 `tilelang_cuda_without_compile` 免除冗余 `nvcc` 编译耗时 |
| `tile-ai/tilelang@256791f` (`#3463`) | `testing/python/language/test_tilelang_language_ldst.py` | TileLang CUDA LDGSTG | 嵌套 128-bit `LDG128`/`STG128` 测试显式校验 `is_cuda_FullySupported()` |

---

## 3. 核心算法与硬件执行机制剖析

### 3.1 PDL（Programmatic Dependent Launch）跨内核流水线重叠

在 DeepSeek-V3 / Kimi-K2 解码路径中，QKV 投影 GEMM（cuBLAS）紧接 `cake_fused_qk_rope_append`（或 FP8 量化版 `cake_fused_qk_rope_fp8_append`）。传统 CUDA 流序要求前序 GEMM 的所有 CTA 完全退场后才启动后续内核；启用 `enable_pdl=True` 后：

1. 主机端通过 `cudaLaunchAttributeProgrammaticStreamSerialization` 启动 `fused_qk_rope_append_kernel<..., /*EnablePDL=*/true>`；
2. 硬件在 QKV GEMM 触发 `cudaTriggerProgrammaticLaunchCompletion` 时即开始在空闲 SM 上预取 RoPE 正余弦表、RMSNorm 权重及分页 KV 页表索引（`indptr`/`indices`）；
3. 在触达依赖 QKV 输出张量 `qkv_input` 的全局内存读取前，调用 `cudaGridDependencySynchronize()` 确保前序写入在 L2 Cache 中全局可见，并在完成自身关键写入后立即发出 `cudaTriggerProgrammaticLaunchCompletion()` 供下游 MLA 注意力内核衔接。

### 3.2 Causal Conv1D 反向传播 `bwd_grid_mult` 尾波切分模型

当训练/微调流与 1–32 个后台通信 CTA 共享同一 GPU 时，若反向传播内核恰好按整机器 SM 数（如 B200 的 148 SM 或 R200 的 SM 数）发射 `grid.x = G` 个长耗时代价 CTA，则被旁路流挤占的最后 1–2 个 CTA 会被迫落入第 2 波次（Wave 2），导致整体耗时翻倍（$T_{\text{total}} \approx 2 \times T_{\text{CTA}}$）。

`d29e16a` 引入 `bwd_grid_mult = k \in \{1, 2, 3, 4\}`：
- 将 token 轴块大小缩小为 $\lceil \text{tokens\_per\_cta} / k \rceil$，总网格扩展为 $G \cdot k$；
- 每个 `dw` 部分和缓冲区槽位仍由 `blockIdx.x % G` 唯一归属并跨子块累加；
- 最终规约屏障等待全部 $G \cdot k$ 个块完成后再写出 `dw`，从而保证 `dw` 累加顺序和 `dx` 输出与 $k=1$ 逐比特完全一致，而单波次溢出代价从 $+100\%$ 降至 $+1/k$（$k=4$ 时仅 $+25\%$）。

---

## 4. 可复现实验验证（CPU 标准库模型）

运行以下命令可复现本阶段 Track 1 的全部定量校验：

```powershell
python experiments\flashinfer_mqa_pdl_tinygemm2_conv1d_mhc_triton_tilelang_phase158\run.py --check
```

关键校验断言：
- `mqa_parallel_schedule.median_speedup == 1.0563` 且 `bitwise_equal_rows == 106`
- `pdl_qk_rope_append.max_fp8_speedup == 1.3558` 且 `max_bf16_speedup == 1.3358`
- `sm107a_tinygemm2.route_9_8192_2048 == "PACKED"` 且 `route_16_8192_16384 == "PACKED_M64_N16"`
- `conv1d_bwd_grid_mult.R200_S261393_C512_1thread_side.speedup == 1.43`
- `compiler_and_graph_guards.mhc_uninitialized_ticket_hazard_rate == 0.0`
