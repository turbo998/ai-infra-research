# FlashInfer Cake Gated SwiGLU MXFP8（SM100/SM103/SM107）、BF16 Rank-Major EP8 MegaMoE Arrive-Only 邮箱、Warp-Decode 调度重调优、Router GEMM 与 Fused Norm Combine Lag-3 流水及 TileLang 架构目标/调优缓存契约

> **文档状态**：Phase 77（`priority: 86`）第一手源码审计 + CPU/offline 合成验证  
> **面向读者**：熟悉云架构、正在深挖 NVIDIA Blackwell（`sm_100a` B200 / `sm_103a` GB300）与 Rubin（`sm_107a` R200）内核生成、MoE 路由/解码流水与 TileLang DSL 编译链的解决方案架构师  
> **English Companion**: [flashinfer-gated-act-mxfp8-ep8-megamoe-warp-decode-norm-combine-tilelang.en.md](flashinfer-gated-act-mxfp8-ep8-megamoe-warp-decode-norm-combine-tilelang.en.md)

---

## 1. 背景与架构问题定义

在大规模 MoE 训练与低延迟解码管线中，激活量化、单节点 8 卡 Expert Parallel（EP8）集合通信、小批量 Routed Fused-MoE Decode 以及 DSL 编译器跨主机交叉编译存在以下五个紧耦合瓶颈：

1. **SwiGLU 激活 + MXFP8 双方向（Rowwise + Colwise）量化的带宽放大**：在 MXFP8 训练/预填充路径中，前向 `silu_and_mul_mxfp8_quantize`（`[M, 2K] -> [M, K]`）与反向 `silu_and_mul_mxfp8_quantize_backward`（`[M, K] -> [M, 2K]`）不仅需要输出 E4M3 数据，还需要以 32 元素为块计算 RCEIL E8M0 缩放因子并写入 `128x4` swizzled TMA 布局。若按固定分块调度，在 `sm_100a`（B200, 148 SMs）、`sm_103a`（GB300, 152 SMs）与 `sm_107a`（R200, 216 SMs）之间无法兼顾小 shape 低延迟与大 shape HBM 带宽饱和。
2. **EP8 MegaMoE 单节点多卡 Barrier 与小 Token 路由的 Launch 开销**：`MoEEpLayer` 在 8×B200（`H=7168, I=2048, E=256`，每卡 32 专家，`top_k=8`）上执行 `1..128` tokens/rank 解码时，原先将输入 staging copy 拆为 3 个独立 kernel，且 input/partial 阶段采用全量栅栏等待全部 peer，同时在 `M <= 16` 时仍单独启动路由 kernel（总计 12 个 kernel）。
3. **NVFP4 W4A4 Routed Fused-MoE Decode（`cake_warp_decode`）与 Router GEMM（`cake_router_gemm`）的按形状调度失配**：在 `1..32` tokens 的解码区间内，不同 `(E, I, top_k, M)` 组合对 device-workfeed 与输出清零（memset vs read）策略的最优切换点不同；而 `cake_router_gemm`（`out[M, N] = A[M, K] · B[N, K]^T`）若固定采用每专家 1 个 CTA，会在 `M=1..16` 时重复从 L2/HBM 读取输入激活 `A[M, K]`。
4. **Fused Norm Combine（`cake_fused_norm_combine`）在 `tokens >= 480` 时的跨卡归约访存延迟**：8 peers、双 norm 轨道（`H=2560`）在长序列预填充尾部归约时，若加载与规约步距过紧，会暴露 NVLink/HBM 读延迟。
5. **TileLang 在无 GPU 构建机上的静默 `sm_50` 降级与 Autotuner 缓存键碰撞**：在 CPU 构建容器上未指定 `arch` 时，`tilelang/cuda/target.py` 原先回退为裸字符串 `"cuda"`，导致底层 TVM 静默填充 `arch=sm_50` 并生成错误的指令序列；同时 `AutoTuner.get_tunner()` 未将自定义 `warmup`、`rep`、`timeout` 透传给 `set_profile_args()`，使 `hash(profile_args)` 在不同评测预算下发生缓存键碰撞。

---

## 2. 第一手源码审计与实现机制

### 2.1 `flashinfer@bfe3de571c09`（`#7305`）：`cake_gated_act_mxfp8_backend` 全 6 形态跨代内核

在 `flashinfer/activation.py` 与 `csrc/cake_gated_act_mxfp8/` 中，`flashinfer.cake_gated_act_mxfp8_backend` 为前向（`[M, 2K] -> [M, K]`）与反向（`[M, K] -> [M, 2K]`）的 3 种量化输出模式（`rowwise`、`colwise`、`both`）共计 6 种算子形态提供原生预编译内核：

- **硬件代际覆盖**：同时编译覆盖 `sm_100a`（NVIDIA B200, 148 SMs）、`sm_103a`（NVIDIA GB300, 152 SMs）与 `sm_107a`（NVIDIA R200, 216 SMs）。
- **按 `M * K` 动态选型与严格逐比特等价**：主机侧根据 `M * K` 总元素规模选择最优分块程序；每個 32 元素块采用 `RCEIL` 方向计算 E8M0 指数缩放并直接按 `128x4` swizzle 写入全局内存，且输出与 FlashInfer 基线逐比特完全一致（bitwise-identical）。
- **作者实测加速比（Author-Benchmark，单卡中位数延迟）**：
  - **B200 (`sm_100a`)**：整体 **`1.075x–1.501x`**（前向 **`1.196x–1.501x`**，如 `128x128` fwd colwise `3.53 -> 2.35 us` **`1.501x`**，`131072x8192` fwd colwise `1117.21 -> 829.95 us` `1.346x`；反向 **`1.075x–1.474x`**，如 `128x128` bwd colwise `3.71 -> 2.52 us` **`1.474x`**，`131072x8192` bwd colwise `1727.04 -> 1374.90 us` `1.256x`）。
  - **GB300 (`sm_103a`)**：整体 **`1.044x–1.411x`**（前向 **`1.169x–1.411x`**，反向 **`1.044x–1.383x`**）。
  - **R200 (`sm_107a`, 216 SMs)**：整体 **`1.094x–1.697x`**（前向 **`1.331x–1.697x`**，如 `131072x8192` fwd colwise `864.60 -> 509.53 us` **`1.697x`**，`16384x7168` fwd colwise `99.37 -> 60.56 us` `1.641x`；反向 **`1.094x–1.427x`**）。

### 2.2 `flashinfer@f911858bfc32`（`#7300`）：BF16 Rank-Major EP8 MegaMoE Arrive-Only 邮箱与内联路由

针对 8×B200（`sm_100a`）上的 `MoEEpLayer`（BF16 rank-major，`H=7168, I=2048, E=256`，8 ranks 各持 32 专家，`top_k=8`），PR `#7300` 实施了三项关键手术：

1. **三合一输入 Staging Copy**：将原先 3 次独立的输入 staging 拷贝合并为 1 个 kernel，直接消除 2 次 kernel launch 与尾块气泡。
2. **1-Warp Arrive-Only 对端邮箱（Per-Peer Mailbox）**：将 input barrier 与 partial barrier 改造为仅含 1 个 warp 的 arrive-only kernel，向目标对端释放独立 mailbox 标志；下游 `dispatch` 与 `combine` 仅等待其实际需要读取的 peer mailbox，避免全局同步木桶效应。
3. **`M <= 16` 时 Dispatch 内联路由计算**：当每卡 token 数 `M <= 16` 时，直接在 `dispatch` 内核中完成 top-k 路由计算，跳过独立的 route kernel，使总启动内核数降至 **9 个**。
- **作者实测加速比（8×B200）**：Eager 模式获 **`1.03x–1.06x`** 加速（`M=1` 全 rank 负载 `173.1 -> 162.8 us` **`1.06x`**，`M=128` `533.6 -> 513.0 us` `1.04x`）；CUDA Graph 模式获 **`1.03x–1.08x`** 加速（`M=1` 全 rank `165.9 -> 153.8 us` **`1.08x`**，`M=1` 单 rank0 `160.7 -> 149.3 us` **`1.08x`**）。

### 2.3 `flashinfer@ba24b4214024`（`#7292`）、`7b220a67a486`（`#7287`）、`dc39352926e8`（`#7285`）与 `2866b59272d7`（`#7311`）

- **`cake_warp_decode` 调度表重调优（`#7292`）**：针对 NVFP4 W4A4 routed fused-MoE decode（`1..32` tokens），在 `sm_100a`（GB200）上重路由 **75 行** `(shape, num_tokens)` 调度配置，最高获 **`1.289x`（`1.29x`）** 加速（`e256_i1536_k8` 在 `M=20` 时 read 路径 `161.68 -> 125.43 us`，ratio `0.776`；`M=32` 时 `158.01 -> 126.16 us`，`1.252x`）；在 `sm_103a`（GB300）上重路由 **26 行**，最高获 **`1.237x`（`1.24x`）** 加速（`e256_i1536_k8` 在 `M=1` 时 `24.02 -> 19.42 us`，ratio `0.809`）。
- **`cake_router_gemm` 按形状专属调度（`#7287`）**：为 `M in 1..16`、`K in {6144, 7168}` 的全部 32 个程序引入跨多专家复用已加载 `A` 分块并在更快时沿 thread-block cluster 切分 `K` 的专属调度。在 96 行评测矩阵中全面超越 1-CTA-per-expert 基线：GB200（`sm_100a`）最高达 **`1.694x`**（`K=7168, N=128, BF16, M=16, PDL=on`：`6.720 -> 3.968 us`），GB300（`sm_103a`）最高达 **`1.733x`**（`6.656 -> 3.840 us`）。
- **`cake_fused_norm_combine` Lag-3 深度软件流水（`#7285`）与 `cake_blackwell_bf16_fp4` M128 Pair PDL（`#7311`）**：在 `tokens >= 480` 的 owner 程序中采用 3 步滞后（lag 3）流水并将最终写出推迟 4 步（`1024` tokens 在 8×B200 上 `30.656 -> 30.112 us` **`1.018x`**，8×B300 上 `31.040 -> 30.560 us` **`1.016x`**）；`#7311` 重新生成大 M `tcgen05` M128 pair PDL 程序，在 GB300 上 `768/2112/2048` tiled 达 **`1.029x`**（`10.860 -> 10.552 us`）。

### 2.4 `tilelang@f1a34efea68c`（`#2860`）与 `90f8908b5f17`（`#3337`）：编译器目标契约与调优器缓存键隔离

- **拒绝无架构裸 `"cuda"` 目标（`#2860`）**：`_cuda_target_from_arch(arch: str | None) -> Target` 在 `arch is None`（无本地 GPU 且未显式指定架构）时立即抛出 `ValueError`，提示调用方显式固定 `target={'kind': 'cuda', 'arch': 'sm_90a'}` 或设置 `TILELANG_DEFAULT_TARGET`，彻底杜绝 TVM 静默回退到 `sm_50` 导致的误编译。
- **Autotuner 评测参数透传（`#3337`）**：`AutoTuner.get_tunner()` 将 `warmup=self.warmup, rep=self.rep, timeout=self.timeout` 显式传入 `.set_profile_args(...)`，保证 `hash(profile_args)` 在不同评测迭代次数或超时阈值下生成独立缓存键。

---

## 3. 关键公式、数据布局与合成实验验证

### 3.1 流量与加速比公式

对于 `silu_and_mul_mxfp8_quantize` 前向内核（输入 `X` 为 BF16 `[M, 2K]`，输出 `Y` 为 FP8 E4M3 `[M, K]` 及每 32 元素 1 字节的 E8M0 缩放因子）：

$$\text{Bytes}_{\text{fwd, single-mode}}(M, K) = 4MK + MK + \frac{MK}{32} = 5.03125 \cdot MK \text{ bytes}$$

$$\text{Bytes}_{\text{bwd, single-mode}}(M, K) = 2MK + 4MK + 2MK + \frac{2MK}{32} = 8.0625 \cdot MK \text{ bytes}$$

本项目合成验证程序 `experiments/flashinfer_gated_act_mxfp8_ep8_megamoe_warp_decode_norm_combine_tilelang/run.py` 对上述 6 个上游提交的定量契约进行了完整复现与不变量校验：

| 子模块 / 算子 | 架构 / 规模 | 基线延迟 ($\mu\text{s}$) | 优化后延迟 ($\mu\text{s}$) | 实测加速比 / 收益 | 证据性质 |
|---|---|---:|---:|---|---|
| `cake_gated_act_mxfp8` (fwd colwise) | `sm_100a` B200, `128x128` | `3.53` | `2.35` | **`1.501x`** | Author-benchmark (`#7305`) |
| `cake_gated_act_mxfp8` (fwd colwise) | `sm_107a` R200, `131072x8192` | `864.60` | `509.53` | **`1.697x`** | Author-benchmark (`#7305`) |
| `MoEEpLayer` EP8 MegaMoE (CUDA Graph) | 8×B200, `M=1` all-ranks | `165.9` | `153.8` | **`1.08x`**（9 vs 12 kernels） | Author-benchmark (`#7300`) |
| `cake_warp_decode` (NVFP4 W4A4) | `sm_100a` GB200, `e256_i1536_k8`, `M=20` | `161.68` | `125.43` | **`1.289x`**（ratio `0.776`） | Author-benchmark (`#7292`) |
| `cake_router_gemm` (`K=7168, N=128`) | `sm_103a` GB300, BF16, `M=16`, PDL=on | `6.656` | `3.840` | **`1.733x`** | Author-benchmark (`#7287`) |
| `cake_fused_norm_combine` (lag-3) | 8×B200, `H=2560`, `1024` tokens | `30.656` | `30.112` | **`1.018x`** | Author-benchmark (`#7285`) |

---

## 4. 架构师设计练习与后续可证伪问题

1. **设计计算练习**：假设某 MoE 层每步执行 1 次 `cake_router_gemm`（`K=7168, N=128, M=16`）、1 次 `MoEEpLayer` EP8 MegaMoE（`M=16`，CUDA Graph）以及 1 次 `cake_fused_norm_combine`（`1024` tokens）。请基于表中数据计算在 B200/GB200 上单层端到端延迟节省（`us`），并评估当 `M` 从 `16` 增至 `17` 时因退出内联路由（9 -> 10 kernels）带来的延迟阶跃。
2. **可证伪问题（Falsifiable Hypothesis）**：
   - **H1**：在 `MoEEpLayer` EP8 MegaMoE 中，若将内联路由阈值从 `M <= 16` 放宽至 `M <= 32`，`dispatch` 内核内部的寄存器压力与 CTA 驻留率下降是否会抵消省去独立 route kernel 的 launch 收益？
   - **H2**：在 `sm_107a`（R200, 216 SMs）上，`cake_gated_act_mxfp8` 在 `both` 模式（同时写 rowwise 与 colwise 张量）下的加速比是否仍受限于 HBM 写带宽而非 SFU 激活吞吐？
