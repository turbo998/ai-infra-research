# FlashInfer Blackwell BF16×FP4 宽松簇屏障与 8-CTA Split-K 归约、W4A16 MegaMoE EP 内核、延迟采样归约及 CuTeDSL / TileLang 同步修复（Phase 103）

- **问题 ID**：`flashinfer-blackwell-bf16-fp4-w4a16-megamoe-cutlass-tilelang-phase103`（`priority: 112`，组件：`gpu-software-stack`）
- **证据等级**：`own-offline`（基于上游源码补丁审计 + Python 3.12 标准库确定性算例 `experiments/flashinfer_blackwell_bf16_fp4_w4a16_megamoe_phase103/run.py`；作者在 B200/B300/GB300 上的 CUPTI 基准记为 `author-benchmark`）
- **配套英文版**：[`flashinfer-blackwell-bf16-fp4-w4a16-megamoe-cutlass-tilelang-phase103.en.md`](flashinfer-blackwell-bf16-fp4-w4a16-megamoe-cutlass-tilelang-phase103.en.md)

## 1. 一手上游信源（零重叠）

1. **FlashInfer `535c18c714` (#7463)**：`perf(cake_blackwell_bf16_fp4): relaxed tcgen05 entry cluster arrive and a leaner 8-CTA split-K reduce-scatter cell`
2. **FlashInfer `124955a41b` (#7557)**：`feat(cake_mega_moe): BF16 × NVFP4 W4A16 MegaMoE EP backend for sm_100a/sm_103a`
3. **FlashInfer `7c87948b77` (#7515)**：`perf(cake_fused_qk_rope_fp8_append): regenerate fold, scan and page-prefetch stages (1.1-1.8 % geomean speedup)`
4. **FlashInfer `cbe168b01e` (#5196)**：`perf(sampling): defer logits block reduction`
5. **FlashInfer `f7d1cb9f55` (#7563)**：`perf(cake_kda): faster prepared BF16-input KDA prefill with FP32 state`
6. **FlashInfer `bbee0f77a0` (#7555)**：`perf(cake_gdn): faster backward prep, dqkg and adjoint-walk tape kernels, forward prep prefetch, two CTAs per SM for backward prep on SM100 and SM103`
7. **CUTLASS `6278db36ee` (#3725)**：`[CuTeDSL] fix: accept bare int as local_partition's index argument`
8. **CUTLASS `af3d4367a8` (#3639)**：`[CuTeDSL] Select CUDA 12.9 grouped GEMM codegen by CTA mode`
9. **TileLang `a765bb4bf2` (#3356)**：`[BugFix][Transform] Unroll the epilogue of a pipeline with a runtime trip count`
10. **TileLang `5434f0ac54` (#3346)**：`[BugFix] Treat an address_of pointer access as a write in ThreadSync`

---

## 2. 核心实现机制与定量分析

### 2.1 Blackwell `cake_blackwell_bf16_fp4` 宽松入口簇屏障与 8-CTA Split-K 16.0x DSMEM 削减 (`535c18c714`)

FlashInfer 的 `mm_bf16_fp4` 将 BF16 激活与带 E4M3 块缩放的 FP4 (`E2M1`) 权重相乘（FP32 累加），提供 `blackwell-native`（cuDNN-ABI 布局）与 `blackwell-tiled`（CuTe-ABI 预排块布局）两套后端共 97 个 Cake 程序。该补丁对其中 9 个关键程序进行两项硬件级优化：
1. **宽松入口簇屏障 (`barrier.cluster.arrive.relaxed.aligned`)**：在 `768/2112/2048` 的 3 个双 CTA `M=128` `tcgen05` 程序中，`fence.mbarrier_init.release.cluster;` 已经完成了对集群内共享内存 `mbarrier` 初始化的发布（Release）。此前紧随其后的 `barrier.cluster.arrive.release.aligned;` 会触发全 GPU 范围的内存写排空（Memory Drain），阻塞各 Warp 进入角色分派；改为 `barrier.cluster.arrive.relaxed.aligned;` 后在 GB300 / B200 上取得 **1.016x / 1.015x** 加速（`8.692 → 8.556 µs` / `9.368 → 9.232 µs`）。
2. **8-CTA Split-K Reduce-Scatter 瘦身与预发 TMA (`7,168 B → 448 B`)**：在 `M=1, N=4096, K=4096` 的 2 个 `blackwell-tiled` 8-CTA Split-K Reduce-Scatter 程序中，TMA 异步加载被提前到启动簇屏障之前发起；同时在 `M=1` 下每个 CTA 仅向 7 个对端 CTA 发送其负责落盘的 1 行输出（每对端 `64 B`，而非此前的 `1,024 B`），使 `mbarrier_expect_tx(smem + 16, ...)` 期望字节数从 $7 \times 1024 = 7{,}168\text{ B}$ 降至 $7 \times 64 = 448\text{ B}$（**16.0x DSMEM 流量削减**），GB300 / B200 延迟分别从 `6.608 → 6.232 µs`（**1.060x**）和 `6.784 → 6.406 µs`（**1.059x**）。

### 2.2 `sm103_bf16_nvfp4_bf16_cake` W4A16 MegaMoE EP 内核与批大小交叉点 (`124955a41b`)

新增第四种 Cake 专家并行 MegaMoE 后端 `sm103_bf16_nvfp4_bf16_cake`（支持 `sm_100a` 与 `sm_103a`），在单内核中融合 Token Dispatch、带 SwiGLU 门控的 FC1、FC2（将预量化 E2M1 + E4M3 块 16 缩放的 NVFP4 权重在线解码为 BF16 送入 Tensor Core MMA，FP32 累加）并通过 PDL 链式启动固定槽位序的 Top-8 Combine 内核。在 NVIDIA B300 (`sm_103a`) 上对比 FlashInfer 自调优 `sm100_bf16_nvfp4_bf16_cutedsl` 基线：
- **单卡 Preset D (`H=7168, I=2048, E=256, top-8, 1 GPU`)**：在单 Token 解码 `T=1` 下从 `103.2 µs` 降至 `64.9 µs`（**1.5912x 加速**），在 `T=8..512` 保持 `1.0204x–1.0558x` 优势；但在 `T=4096` 大批 Prefill 下为 `3178.5 µs` vs 基线 `2968.5 µs`（比值 `0.9339x`）。
- **8 卡 EP8 Preset G (`H=6144, I=2048, E=256, top-8, EP8`)**：在 `T=4096` 下从 `415.2 µs` 降至 `383.0 µs`（**1.0839x 加速**），而在中等批大小 `T=8..512` 下比值为 `0.9136x–0.9377x`。
- **架构师路由启示**：W4A16 MegaMoE 存在明显的批大小与 EP 拓扑交叉点——运行时应按 `(EP_size, T)` 实施双内核自适应派发（单卡 `T <= 512` 及 EP8 `T >= 4096` 选 Cake，其余选 CuTeDSL 自调优），在 Preset D 与 Preset G 全长轨迹上分别合成至 `6,164.5 µs` 与 `1,228.2 µs` 最优总耗时。

### 2.3 `SamplingFromLogitsKernel` 词表分块延迟归约 (`cbe168b01e`)

此前 `SamplingFromLogitsKernel` 对每个词表分块（每块 $1024 \times 4 = 4{,}096$ 个元素）执行一次 `BlockReduce` 与块级屏障。在 Qwen 等 `V = 151,936` 的 FP32 词表上，每次采样需执行 **38 次**块级归约与同步。补丁改为每个线程在寄存器内跨 38 个词表分块维护局部最优候选，循环结束后仅执行 **1 次** `BlockReduce`（**38.0x 块归约削减**）。为保证与原多轮 CUB 左折叠完全一致的确定性平局决胜（Tie-breaking），相等数值统一取更大词表下标（`DataAndIndex::operator+` 保留右操作数），确定性算例验证在 4 个跨块同分位置 `[511, 4113, 78127, 151594]` 上单次归约与逐块归约均精确选中 `151594`。

### 2.4 CUTLASS CuTeDSL 与 TileLang 编译器正确性修复 (`6278db36ee`, `af3d4367a8`, `a765bb4bf2`, `5434f0ac54`)

- **CuTeDSL `local_partition` 裸 `int` 支持 (`6278db36ee`)**：为 `local_partition` 增加 `elif isinstance(index, int): index_val = const(index, Int32, loc=loc, ip=ip)`，消除传入 Python `int` 时的 `AttributeError: 'int' object has no attribute 'ir_value'`。
- **CuTeDSL CUDA 12.9 分组 GEMM 按 CTA 模式选优 (`af3d4367a8`)**：在 CUDA 12.9 下，2-CTA 指令模式（`use_2cta_instrs=True`）启用向量化 RMEM 与 `-O2`，而 1-CTA 模式保持标量 RMEM 与 `-O3`。
- **TileLang 动态循环计数流水线尾声展开 (`a765bb4bf2`)**：当流水线主循环次数 `n` 为运行时变量、流水线级数为 `s` 时，尾声循环跨度为 `(n + s) - n`；调用 `analyzer_.Simplify(extent)` 将其化简为常量 `s` 后再展开，避免未展开尾声向 `ptx_wait_group` 传入非法运行时变量。
- **TileLang `ThreadSync` 指针取址写冲突建模 (`5434f0ac54`)**：`address_of` 将共享内存指针传给外部不透明函数时，旧分类器仅记为 `kRead`，导致线程 0 通过外部函数清零共享内存后、全块读取前缺失 `__syncthreads()`；补丁同时记录 `kRead` 与 `kWrite`，将插入屏障数从 `0` 恢复为 `1`。

---

## 3. 架构师设计练习与可证伪问题

1. **练习计算**：在 8-CTA Split-K Reduce-Scatter 中，若输出行数从 `M=1` 增至 `M=4`（每行 32 个 BF16 元素 = `64 B`），每个 CTA 仅向每个对端发送对应落盘行时，`mbarrier_expect_tx` 的期望字节数是多少？（答案：$7 \times 4 \times 64\text{ B} = 1{,}792\text{ B}$，仍比旧版固定 $7{,}168\text{ B}$ 小 **4.0x**）。
2. **可证伪假设**：若在 `SamplingFromLogitsKernel` 中将平局决胜规则改为保留较小下标（左操作数优先），当两个不同线程分块内出现相同最大带噪 Logit 时，延迟单次归约与逐块归约仍将保持一致，但会破坏与历史版本保留最大下标的回归兼容性。
