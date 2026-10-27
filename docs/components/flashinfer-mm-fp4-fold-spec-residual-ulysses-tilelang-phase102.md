# FlashInfer `cake_mm_fp4` 簇拆分与余数块折叠、`ChainSpeculativeSampling` 延迟残差归约、Cake Ulysses World-2 直连 All-to-All 与 TileLang TMA 流绑定（Phase 102）

- **问题 ID**：`flashinfer-mm-fp4-fold-spec-residual-ulysses-tilelang-phase102`（`priority: 111`，组件：`gpu-software-stack`）
- **证据等级**：`own-offline`（基于上游源码补丁审计 + Python 3.12 标准库确定性算例 `experiments/flashinfer_mm_fp4_fold_spec_res_ulysses_phase102/run.py`）
- **配套英文版**：[`flashinfer-mm-fp4-fold-spec-residual-ulysses-tilelang-phase102.en.md`](flashinfer-mm-fp4-fold-spec-residual-ulysses-tilelang-phase102.en.md)

## 1. 一手上游信源（零重叠）

1. **FlashInfer `cb138f8aa6` (#7550)**：`perf(cake_mm_fp4): cluster-split per-token quantizer and token-remainder GEMM folds on B200 and GB300`
2. **FlashInfer `03bd5b3402` (#5201)**：`perf(sampling): defer chain speculative residual reduction`
3. **FlashInfer `88205170de` (#7566)**：`perf(cake_ulysses): faster world-2 direct all-to-all, symmetric handshake`
4. **FlashInfer `62b5ae4e52` (#7336)**：`fix(ffi): represent absent functions and MoE tensors explicitly`
5. **FlashInfer `8e7a21d538` (#7565)**：`test(cake_gdn): collect garbage before the GDN prefill allocation-count window`
6. **NVIDIA CUTLASS `c8fc8a6535` (#3666)**：`[CuTeDSL] docs: fix cutlass.range unroll=0 and unroll_full descriptions`
7. **NVIDIA CUTLASS `ee98dbdfd5` (#3693)**：`Keep plain pytest functions in separate sharding groups`
8. **NVIDIA CUTLASS `218d3bfc4a` (#3720)**：`[CuTeDSL] docs: fix local_tile example output shapes`
9. **TileLang `837486abf7` (#3365)**：`Fix constant row indices for TMA gather and scatter`
10. **TileLang `0d75cd9af5` (#3359)**：`[BugFix][CUDA] Honor tensor-device streams in NVRTC and CuTeDSL`

---

## 2. 核心实现机制与定量分析

### 2.1 FlashInfer `cake_mm_fp4` 簇拆分激活量化与余数 Token GEMM 折叠 (`cb138f8aa6`)

在 Blackwell（`sm_100a` B200 / `sm_103a` GB300）上，`cake_mm_fp4` 执行逐 Token NVFP4 激活量化（block-16 scale）与 NVFP4×NVFP4 GEMM：
- **独立量化器 2/4/8-CTA Cluster Split (`QUANT_ALONE_CLUSTER_SPLIT`)**：当独立调用 `nvfp4_quantize(..., per_token_activation=True, backend="cake")` 时，宽行（`K ∈ {16384, 18432, 28672}`）在小 Token 批次（`M=1..8` 或 `M=130`）下若仅用单 CTA 处理一整行会严重受限于 SM 占用率。补丁按架构与 `(low, high, C)` 将单行拆分为 `C ∈ {2, 4, 8}` 个 CTA 的 Cluster 并行归约：
  - B200 (`sm_100a`) 14 个测试点加速 **1.027x–1.085x**（几何平均 **1.0579x**，峰值 `K=28672, M=1` `2.653 → 2.454 µs`）；
  - GB300 (`sm_103a`) 12 个测试点加速 **1.050x–1.086x**（几何平均 **1.0649x**，峰值 `K=16384, M=1` `2.381 → 2.190 µs`）。
- **Token-Remainder GEMM 边界折叠 (`K×N = 7168×18432`)**：
  - 在 `M ∈ [129, 136]` 时，旧路由启动 2-CTA 256-Token pair tile（近半数行为空填充）；新路由将 `1..8` 行余数直接折叠进单个 **1-CTA 128-Token tile**（链式模式附带入口 L2 权重预取），B200/GB300 加速 **1.028x–1.059x**；
  - 在 `M ∈ [257, 272]` 时，旧路由启动两个 2-CTA 256-Token pair tile（第二个 tile 仅承载 `1..16` 行有效 Token）；新路由将余数折叠进单个 **2-CTA 256-Token pair tile**，直接消除整波尾块：B200 GEMM 加速 **1.233x**（`25.91 → 21.00 µs`），GB300 GEMM 加速 **1.194x**（`23.82 → 19.94 µs`）。

### 2.2 `ChainSpeculativeSampling` 延迟残差归约 (`03bd5b3402`)

链式投机采样从残差分布 $\max(q_i - p_i, 0)$ 中采样首个拒绝位置。主干代码在遍历词表分块循环（步长 $\text{BLOCK\_THREADS} \times \text{VEC\_SIZE} = 1024 \times 4 = 4096$）的每一轮迭代内都调用了 `BlockReduce::Sum` 与 `__syncthreads()`，而循环体内部根本不读取累加和 `sum_relu_q_minus_p`。
- **优化公式**：每个线程在寄存器变量 `threadlocal_sum_relu_q_minus_p` 中跨词表分块累加，循环结束后仅执行 **1 次** `BlockReduce`：
  $$\text{Barriers}(V=128{,}256) : \left\lceil \frac{128{,}256}{4096} \right\rceil = 32 \longrightarrow 1 \quad (\mathbf{32.0\times\ \text{栅栏削减}})$$
- **实测收益**：在全部 32 组 `(Batch, Vocab, Speculate)` 配置上，A100-80GB 取得 **1.13x–1.66x** 加速（平均延迟降低 **22.5%**，`B=1, V=128256, Spec=3` 从 `47.72 → 28.67 µs`），H100-80GB 取得 **1.05x–1.65x** 加速（平均延迟降低 **20.1%**，`34.29 → 20.80 µs`）。

### 2.3 Cake Ulysses World-2 直连 All-to-All 对称握手与 TileLang TMA/Stream 修复 (`88205170de`, `837486abf7`, `0d75cd9af5`)

- **Cake Ulysses Direct Route (`88205170de`)**：更新全部 24 个直写注册输出（`allocate_output`）的 NVLink sequence-parallel All-to-All 内核。在 `world_size=2` 下，由 **128 个 CTA** 流式直写对端显存分块，同时 **20 个 CTA** 从启动时刻并发拷贝本地分块，并移除 `if (out_chunks > 0)` 对 entry/landed 跨卡 `ld.acquire.sys.global.u32` 握手的条件分支依赖。在 `seq=32760, heads=40, head_dim=128` 上，B200 Scatter/Gather 加速 **1.039x / 1.031x**（`134.04 → 128.98 µs`），B300 加速 **1.030x / 1.027x**（`134.45 → 130.50 µs`）。
- **TileLang TMA Gather4/Scatter4 常量行索引与多卡 Stream 绑定 (`837486abf7`, `0d75cd9af5`)**：将 Python `int` 行索引自动归一化为 `tirx.const(row, dtype="int32")`；并在 NVRTC 与 CuTeDSL JIT 适配器中使用 `torch.cuda.current_stream(first_tensor.device)` 替代无参 `current_stream()`，避免多 GPU 进程在非 0 号卡张量上错误取用 `cuda:0` 默认流。

---

## 3. 架构师设计练习与可证伪问题

1. **练习计算**：当 `V = 151,937`（Qwen 类词表）、`BLOCK_THREADS = 1024`、`VEC_SIZE = 4` 时，`ChainSpeculativeSampling` 延迟归约可消除多少次 `BlockReduce` + `__syncthreads()`？（答案：$\lceil 151937 / 4096 \rceil - 1 = 38 - 1 = 37$ 次）。
2. **可证伪假设**：在 `M = 273`（超出 `256 + 16 = 272` 折叠上限 1 行）时，`cake_mm_fp4` 会回退至双 256-Token pair tile 路由，此时 GEMM 延迟将出现阶跃回升。
