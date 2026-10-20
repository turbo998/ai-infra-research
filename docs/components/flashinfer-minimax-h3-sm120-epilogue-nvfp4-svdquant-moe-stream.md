# FlashInfer MiniMax-H3 SM120 Epilogue 代理栅栏与 Ping-Pong 屏障闭合、CAKE NVFP4 SVDQuant 15-Program GEMM 后端及 Rubin MoE `execute_on_streams` 适配

> **语言 / Language**: **中文 (当前)** | [English (`flashinfer-minimax-h3-sm120-epilogue-nvfp4-svdquant-moe-stream.en.md`)](flashinfer-minimax-h3-sm120-epilogue-nvfp4-svdquant-moe-stream.en.md)
> **证据等级**: `implementation` + `author-benchmark`（上游源码/提交）+ `own-offline`（本项目标准库确定性复现实验）
> **上游版本锚点**:
> - `flashinfer-ai/flashinfer@aacd0f134f2a35eaea9ea534aa5b5b31534db449` (`2026-10-20`, PR `#6847`)
> - `flashinfer-ai/flashinfer@ec3bcf0301dac6455897d4a450115fdd8fea1bd3` (`2026-10-20`, PR `#6878`)
> - `flashinfer-ai/flashinfer@6445497657b7bc06fcc768d6d499e38627221818` (`2026-10-20`, PR `#6488`)
> - `flashinfer-ai/flashinfer@5ea501832bf7e37a7acde903a32329d580c0e9b4` (`2026-10-20`, PR `#6869`)

---

## 1. 核心架构发现与问题定位

### 1.1 MiniMax-H3 SM120 稠密/变长注意力：Generic/Async 代理栅栏时序与 Ping-Pong 命名屏障闭合 (`aacd0f1`)

在 Blackwell 工作站/边缘架构 (`sm_120a`) 与 GB300 (`sm_103a`) 的 `kernel_minimax_h3_dense_attention` (`csrc/cake_minimax_h3_dense_attention_sm120a.cu`) 中，存在两类隐蔽的硬件级同步缺陷：

1. **跨 Warp Generic/Async Proxy 竞态（Proxy Fence 时序错误）**：
   - 算子在 Epilogue 阶段通过 **Generic Proxy** 将输出 $O$ 暂存（Stage）在复用共享内存缓冲区 `q_smem` 中（同时 Q 缩放也在写该缓冲区，Epilogue 再将其读回写出到全局内存 $O$）。随后，下一个持久化工作项（Work Item）中由 Warp 0 的 `elect_sync()` 线程发起异步 TMA 加载（`mbarrier_arrive_expect_tx(q_full_addr, 32768)`），通过 **Async Proxy** 向同一块 `q_smem` 灌入新 Tile。
   - **修复前缺陷**：内核在工作项末尾仅执行了 `__syncthreads()`，而将 `asm volatile("fence.proxy.async.shared::cta;" ::: "memory");` 放在**下一工作项开头**（`__syncthreads()` 之后）。根据 PTX 内存一致性模型，单线程的 `fence.proxy.async.shared::cta` **仅排序该线程自身**此前发起的 Generic Proxy 访问；Warp 1..7 对 `q_smem` 的 Generic 读写从未在 Warp 0 发起 TMA 重填之前完成跨代理排序，导致每次重填存在 `7` 个 Warp 的跨代理数据竞争风险（8 个工作项共 `49` 次潜在代理冲突）。
   - **修复方案**：移除循环顶部的栅栏，改为在每个工作项末尾、**`__syncthreads()` 之前**由全部 256 个线程执行 `fence.proxy.async.shared::cta;`。这样每个线程先将自身的 Generic Proxy 访问刷入 CTA 共享内存视图，再通过 `__syncthreads()` 向 Warp 0 的 Elected 线程传递可见性，实现跨代理零竞态（`49 -> 0`）。

2. **Ping-Pong 命名屏障未闭合退出（Named Barrier 1 Open Round）**：
   - 在稠密注意力、FP8 变长注意力（`cake_minimax_h3_sm120_quant_varlen_attention_sm120a.cu`）与 NVFP4 变长注意力（`cake_minimax_h3_sm120_nvfp4_varlen_attention_sm120a.cu`）中，两个 Warp Group（Group 0: `warp < 4` 共 128 线程；Group 1: `warp >= 4` 共 128 线程）在命名屏障 `1` 和 `2`（阈值 `256` 线程）上交替 Ping-Pong。
   - 由于 Group 1 在进入持久化循环前对命名屏障 `1` 执行了一次预到达（Pre-arrive），Group 1 在屏障 `1` 上的到达轮次始终领先 Group 0 一轮。当持久化循环结束时，Group 1 的最后一次到达开启了一个新的命名屏障轮次（`128/256` 线程到达），但没有任何 Group 0 的 `barrier.sync` 与之配对，导致 CTA 在命名屏障 `1` 仍处于未完成状态下退出，触发 `compute-sanitizer synccheck` 报错。
   - **修复方案**：在持久化循环退出后、`__syncthreads()` 清理前，由 Group 0 补齐一次闭合同步：
     ```cuda
     if (warp < 4) {
         asm volatile("barrier.sync 1, 256;" ::: "memory");
     }
     ```
   - **验证结论 (`author-benchmark`)**：在 GB300 (`sm_103a`) 上通过 `compute-sanitizer racecheck/synccheck` 零错误验证，耗时波动在 `±0.35%` 以内，稠密内核 SASS 指令数不变。

---

### 1.2 CAKE NVFP4 SVDQuant 15-Program GEMM 后端收敛 (`ec3bcf0`)

`flashinfer.mm_nvfp4_svdquant(..., backend="cake")` 计算融合低秩 LoRA 补偿的 NVFP4 SVDQuant 矩阵乘：
$$\text{out} = \text{BF16}\!\left(\alpha \cdot \left(\text{deq}(A)\,\text{deq}(B)^\top + D\,L_1^\top\right) + \text{bias}\right)$$
其中 $A, B$ 为 NVFP4 (`E2M1`) 配合 `E4M3` 块级缩放因子（`SFA`, `SFB`），低秩矩阵 $D \in \mathbb{R}^{M \times r}$、$L_1 \in \mathbb{R}^{N \times r}$、可选 `bias` 与输出均为 BF16，累加精度为 FP32，覆盖 LoRA 秩 $r \in \{32, 64, 96, 128\}$。

- **架构升级**：PR `#6878` 将原先分散的 7 个 Cake 内核收敛为独立可嵌入包 `flashinfer/cake_nvfp4_svdquant_gemm_backend`，共包含 **15 个 Cake 程序**（覆盖 `sm_100a` B200 与 `sm_103a` GB300、带/不带融合 bias 及 4 种 LoRA 秩，采用 TMEM `80~512` 列分配与 `5~14` 级 TMA 流水）。
- **作者实测性能 (`author-benchmark`, CUDA 13.4, Cold L2 CUPTI Median)**：

| 硬件平台 | 形状 $(M, N, K, r, \text{bias})$ | 旧版 Cake ($\mu\text{s}$) | CUTLASS 最优 ($\mu\text{s}$) | 新版 15-Prog Cake ($\mu\text{s}$) | 相对旧版 Cake 加速 | 相对 CUTLASS 加速 | 算法吞吐 ($\text{TFLOPS/s}$) |
|---|---|---:|---:|---:|---:|---:|---:|
| **B200 (`sm_100a`)** | `(129, 3072, 3072, 32, 1)` | 10.66 | 8.19 | **6.46** | **1.65x** | **1.267x** | 381.20 |
| **B200 (`sm_100a`)** | `(129, 3072, 3072, 32, 0)` | 6.46 | 8.19 | **6.34** | **1.02x** | **1.293x** | 388.42 |
| **B200 (`sm_100a`)** | `(129, 3072, 12288, 128, 1)` | 26.37 | 16.29 | **13.95** | **1.89x** | **1.167x** | 705.29 |
| **B200 (`sm_100a`)** | `(6889, 12288, 3072, 32, 1)` | 319.39 | 103.23 | **97.38** | **3.28x** | **1.060x** | 5407.76 |
| **B200 (`sm_100a`)** | `(16384, 12288, 3072, 32, 1)` | 731.39 | 253.40 | **242.88** | **3.01x** | **1.043x** | 5152.94 |
| **GB300 (`sm_103a`)** | `(129, 3072, 3072, 32, 1)` | 10.46 | 7.20 | **5.98** | **1.75x** | **1.203x** | 411.80 |
| **GB300 (`sm_103a`)** | `(6889, 12288, 3072, 32, 1)` | 293.30 | 85.17 | **83.17** | **3.53x** | **1.024x** | 6331.70 |
| **GB300 (`sm_103a`)** | `(16384, 12288, 3072, 32, 1)` | 673.32 | 195.98 | **187.87** | **3.58x** | **1.043x** | 6661.77 |

---

### 1.3 Rubin 局部性域 MoE `execute_on_streams` 显式输入序列适配 (`6445497`)

- 在 PyTorch PR `#199128` 更新 `torch.cuda.execute_on_streams(streams, fn, *inputs)` 契约后，回调函数不再隐式接收流下标，而是从显式传入的 `*inputs` 序列中解包参数。
- FlashInfer `fused_moe/cute_dsl/fused_moe.py` 在 FC1 与 FC2 的 Rubin 局部性域（Locality-Domain）并发执行路径中显式传入 `range(len(localized_streams))`，移除了51行旧版自定义 `_execute_on_streams` 兼容包装；同时在 `rubin/blockscaled_contiguous_grouped_gemm_finalize_fusion.py` 中将 `cp.reduce.async.bulk` 的 `copy_size = cta_tile_shape_mnk[1] * (c_dtype.width // 8)` 下沉至编译期常量分支（`cta_tile_n=256` 下 BF16 为 `512 B`，FP32 为 `1024 B`）。

---

## 2. 本项目确定性复现实验 (`own-offline`)

- **实验脚本**: [`experiments/flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe/run.py`](../../experiments/flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe/run.py)
- **验证命令**: `python experiments/flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe/run.py --check`
- **不变量验证结果**:
  - `sm120_proxy_hazards_eliminated = true`（`49 -> 0` 跨 Warp 代理冲突）
  - `sm120_ping_pong_barrier1_closed = true`（`1 -> 0` 未闭合命名屏障轮次）
  - `cake_nvfp4_svdquant_15_programs_converged = true`（B200 最高 `3.28x` vs 旧 Cake / `1.293x` vs CUTLASS；GB300 最高 `3.58x` vs 旧 Cake）
  - `rubin_moe_explicit_stream_inputs_verified = true`（`copy_size` 分别为 `512 B` / `1024 B`）
