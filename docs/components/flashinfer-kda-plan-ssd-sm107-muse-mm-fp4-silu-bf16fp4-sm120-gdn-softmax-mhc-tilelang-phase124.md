# FlashInfer KDA Plan 缓存、Rubin SM107 SSD Combined、Muse Spark Norm、MM FP4 2x2 Multicast、Grouped FP8 SiLU CTA-Cap、BF16-FP4 PDL Alpha、SM120 Weight-First、GDN Det Train、Softmax Balanced Chunks、mHC 释放与 TileLang 边界保护深挖（Phase 124）

> **English Edition**: [flashinfer-kda-plan-ssd-sm107-muse-mm-fp4-silu-bf16fp4-sm120-gdn-softmax-mhc-tilelang-phase124.en.md](flashinfer-kda-plan-ssd-sm107-muse-mm-fp4-silu-bf16fp4-sm120-gdn-softmax-mhc-tilelang-phase124.en.md)
> **证据等级**：`implementation` + `author-benchmark` + `own-offline`（零 GPU 硬件重测声明）

---

## 1. 背景与核心架构问题

在 Blackwell (`sm_100a` B200 / GB200、`sm_103a` B300 / GB300、`sm_120a` RTX PRO 6000) 与 Rubin (`sm_107a`) 架构进入大规模生产训练与推理时，算子库与 DSL 编译器面临十个具体的底层瓶颈：

1. **KDA 训练的重复 `plan()` 开销与 Arena 主机同步 (`bd60c11809`)**：`cake_kda_train` 每次调用默认入口都对同一问题重新执行 63 次 `select_record`（耗时 `1.044 ms`），且 `KdaArena(lean=True, tail_passes=4)` 与 `batch_tokens + doc_lengths` 在主机侧构造张量触发隐式 `cudaStreamSynchronize`，在 98 ms GPU 负载后排队等待高达 `101.45–105.88 ms`。
2. **Rubin `sm_107a` Mamba2 SSD Prefill 的 CTA Cluster 调度与单块尾端传递 (`ecd4d71d89`)**：`cake_ssd_combined` 在 `sm_107a` 上使用 2-CTA/4-CTA cluster 会降低部分中等序列形状的 SM 占用率，且 `tail_chunks == 1` 的变长 exact-prefix 缺乏单块标志字（runtime-ticket flag word）直接传递。
3. **GB300 / B200 Gated Residual RMSNorm 与 Next-RMSNorm 路由断档 (`d9447704cb`)**：`cake_muse_spark_norm`（`H=8192`）在 GB300 PDL 图回放与 B200 `M = 1152..4096` 区间缺少 4-CTA 双行 L2 预取与 L2 load-hint 路由。
4. **GB300 FP4 GEMM 在 `M = 1793..2048` 频段的 Scale-Factor 带宽瓶颈 (`c9fc6cab40`)**：`cake_mm_fp4` 在 152 SM 的 GB300 上需要 8-CTA `2x2` 双向多播（`_sfr_mc42`）分摊 NVFP4/MXFP4 缩放因子环路带宽。
5. **Grouped FP8 Fused SiLU Quant 的 128-CTA 硬上限 (`a0384efcba`)**：`FUSED_MIXED_ROUTE` 被固定限制在 128 CTAs（64 个 2-CTA clusters），无法利用 B200 的 148 SMs 或 GB300 的 152 SMs。
6. **Blackwell BF16×FP4 小 M PDL 核函数的非相干 `alpha` 抢跑读取 (`464ca3c813`)**：`16x1024x1024` `blackwell-native` PDL 程序使用只读非相干加载读取 `alpha[0]`，编译器可将其重排至 `griddepcontrol.wait` 之前，导致读取前序核尚未写完的旧值。
7. **SM120 低延迟 MoE 权重加载与激活 Gather 串行 (`1e7a60d232`)**：`frost_low_latency` 在 `H=2048, I=512, top_k=10` 下需通过 `weight_first=True` 将权重 TMA 加载与软件激活 gather 重叠。
8. **确定性 GDN 训练主机分发、Blackwell Softmax 分块不均与 mHC 反向显存滞留 (`4cc027c249`, `f4be6e9df5`, `1fb63709b8`)**：`cake_gdn_det_train` 缺少最长序列优先（LPT）与双流尾填充；`cake_softmax` 强制按 2048 整 tile 切分导致尾 CTA 负载悬殊；`cake_mhc` 反向传播将 `dY` 保留至整网反传结束（单层 `508.2 MiB`，60 层滞留 `29.29 GiB`）。
9. **TileLang `T.Parallel` 类型转换、批处理归约越界写共享内存与 `*_handle` Restrict 丢失 (`714cbf150d`, `dbabf135f9`, `a366af801a`)**：`ReduceLowerer` 在 `thread_offset > 0`（如 GEMM 占 `[0, 128)`、Reduce 占 `[32, 160)`）时未对 `run_batch` 加线程边界守卫，导致 `[0, 32)` 的 32 个非参与线程发生无符号下溢，覆写共享内存工作区前方的 `64` 个 float（`256 B`）。

---

## 2. 核心实现路径与定量证据分析

```
+---------------------------------------------------------------------------------------------------+
| Phase 124 Track A: Blackwell / Rubin Kernel Scheduling & TileLang Shared-Memory Safety Pipeline   |
+---------------------------------------------------------------------------------------------------+
| 1. cake_kda_train (`bd60c11809`)                                                                  |
|    plan() cache key: (S, num_docs, H, sm_count, arch) -> 63 -> 0 select_record calls (0.0016 ms)  |
|    device arange + non_blocking=True pinned doc_lengths -> 0 host syncs (105.88 ms -> 4.33 ms)    |
+---------------------------------------------------------------------------------------------------+
| 2. cake_ssd_combined (`ecd4d71d89`) & cake_mm_fp4 (`c9fc6cab40`) & SiLU CTA-cap (`a0384efcba`)    |
|    - Rubin sm_107a: 20/56 1-CTA cluster (`__cluster_dims__(1,1,1)`) + 3 exactpre_*_tailone (1.084x)|
|    - GB300 mm_fp4: cluster-of-8 2x2 multicast (`_sfr_mc42`) over M=1793..2048 (1.038x-1.066x)     |
|    - Grouped FP8 SiLU Quant: makespan model lifts CTA cap 128 -> {144, 148, 152} (1.064x-1.073x)  |
+---------------------------------------------------------------------------------------------------+
| 3. Coherence, Chunk Balancing & Autograd Lifetime (`464ca3c813`, `f4be6e9df5`, `1fb63709b8`)       |
|    - BF16xFP4 PDL: volatile `ld.global.b32` for alpha[0] strictly after `griddepcontrol.wait`     |
|    - Softmax sm_100a: ceil(vocab/CTAs) rounded to 32 -> 219 shapes 1.045x geomean (up to 1.094x)  |
|    - mHC backward: drop C++ AutogradNode dY reference immediately -> saves 508.2 MiB/layer        |
+---------------------------------------------------------------------------------------------------+
| 4. TileLang Compiler Passes (`714cbf150d`, `dbabf135f9`, `a366af801a`)                            |
|    - T.Parallel: tirx.convert(coalesced_width) -> IntImmNode                                      |
|    - ReduceLowerer: guard run_batch within [thread_offset, thread_offset + extent) (prevents 256B)|
|    - NonRestrictCollector: preserve __restrict__ on *_handle pointer parameters                   |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 `cake_kda_train` 计划缓存与零同步 Arena (`bd60c11809`)
- **主机侧缓存与动态架构选择**：`cake_backend.plan()` 以 `(S, num_docs, H, sm_count, arch)` 为键缓存计划；在 `S = 261,393`（4 篇文档 `120001 + 70707 + 65537 + 5148`）、`H = 4, D = 128` 上，重复问题的 `plan()` 耗时从 `1.044 ms`（63 次 `select_record`）降至 `0.0016 ms`（0 次选择，加速 **652.5x**）。`plan(arch=None)` 自动读取当前 CUDA 设备架构（在 GB300 上正确选择 `sm_103a`，修复了此前硬编码 `sm_100a` 被 `KdaWorkspace` 拒绝的问题）。
- **消除主机同步**：`KdaArena(lean=True, tail_passes=4)` 的尾切分范围计数改由设备端 `arange` 生成，`doc_lengths` 使用 pinned memory 配合 `non_blocking=True` 异步拷贝。每步主机同步次数从 `2–6` 次降至 **`0` 次**；在 98 ms GPU 旋转负载后的主机排队等待时间从 `101.45 ms` / `105.88 ms`（GB200）骤降至 **`1.73 ms` / `4.33 ms`**，且步骤结束后保留显存为 `0 B`（峰值 `6.97 GB` 不变）。

### 2.2 Rubin `sm_107a` SSD Combined 与 GB300 Muse Norm / MM FP4 / SiLU CTA-Cap
- **Rubin `sm_107a` Mamba2 SSD (`ecd4d71d89`)**：在 56 个 `sm_107a` 程序中将 20 个改为 `__cluster_dims__(1, 1, 1)` 单 CTA cluster 启动，新增 3 个 `exactpre_{bf16,f16,f32}_varlen_tailone` 单块尾端程序（通过 runtime-ticket flag word 交接 `tail_chunks == 1`），并在 `exactpre_f32_varlen_claim` 中通过共享内存广播序列表。在 93 个评测形状中改变 45 个，取得 **`1.004x–1.084x`** 加速（45 个变化形状几何平均 **`1.022x`**，如 `f32 B=2 L=4096 varlen` 从 `566.5 us` 降至 `522.5 us`）。
- **GB300 / B200 `cake_muse_spark_norm` (`d9447704cb`)**：针对 `H=8192` 的门控残差 RMSNorm 与 next-RMSNorm 重调路由（79 条记录、147 个库）。GB300 PDL 图回放模式下 `mode 0 M=192` 达 **`1.1360x`**（`3.439 -> 3.027 us`），kernel-alone `mode 0 M=512` 达 **`1.1289x`**（`9.847 -> 8.722 us`）；B200 next-RMSNorm 在 `M=1152..4096` 引入 4-CTA 双行 L2 预取，在 `M=1536` 达 **`1.0478x`**（`28.224 -> 26.936 us`）。
- **GB300 `cake_mm_fp4` 2x2 多播 (`c9fc6cab40`) 与 Grouped FP8 SiLU CTA-Cap (`a0384efcba`)**：
  - `cake_mm_fp4` 新增 8-CTA `2x2` 多播 scale-factor ring 程序（`_sfr_mc42`），在 GB300 `M = 1793..2048` 的 12 个形状上实现 **`1.038x–1.066x`** 加速（`1920x18432x7168` NVFP4×NVFP4→BF16 从 `73.12 us` 降至 `68.61 us`）。
  - `cake_grouped_fp8_fused_silu_quant` 的 `select_route_and_cap` 在 `2H=2048, K=4096` 下根据 makespan 模型将 `FUSED_MIXED_ROUTE` 的 CTA 上限从 128 提升至 `{144, 148, 152}`。在 `M=8192`（16 专家，`640×8 + 384×8`）上切换至 148 CTAs（74 个 2-CTA clusters），B200 graph/eager 加速 **`1.064x` / `1.073x`**（`70.86 -> 66.54 us`），GB300 加速 **`1.032x` / `1.043x`**。

### 2.3 PDL 相干加载、SM120 Weight-First、GDN、Softmax 均衡分块与 mHC 显存释放
- **Blackwell BF16×FP4 PDL `alpha` 相干性修复 (`464ca3c813`)**：将 `16x1024x1024` `blackwell-native` PDL 程序中的 `alpha[0]` 加载替换为 `griddepcontrol.wait` 之后的 volatile `ld.global.b32`，消除编译器重排引发的数据竞争，同时在 GB300/B200 上取得 `1.007x–1.018x` 加速。
- **SM120 Weight-First (`1e7a60d232`) 与确定性 GDN (`4cc027c249`)**：SM120 新增 `fp8-grouped-tc-w8-weight-first-v6` 使权重 TMA 预取与激活 gather 重叠；`cake_gdn_det_train` 通过 LPT 最长序列优先调度与第二 CUDA 流尾轮填充，在 B200 `4x4096 H8/HV16` fwd+bwd 达 **`1.289x`**（`0.8435 -> 0.6544 ms`），在 GB300 `3 seqs = 12,750 tok` fwd 达 **`1.279x`**（`0.2407 -> 0.1882 ms`）。
- **Blackwell Softmax 均衡分块 (`f4be6e9df5`) 与 mHC `dY` 即时释放 (`1fb63709b8`)**：
  - `cake_softmax` 将每个 CTA 的负责元素数从 `2048` 整倍数缩减为 `ceil(ceil(vocab / CTAs) / 32) * 32`：在 B200 全部 219 个分块变化形状上取得 **`1.045x` 几何平均加速**（最高 **`1.094x`**，`67x81928` 从 `12288 -> 10272` 元素/CTA，`10.37 -> 9.50 us`）。
  - `cake_mhc` 在每个连接反传完成后立即清空 C++ `AutogradNode` 持有的 `dY`（`[T, 4, 4096]` BF16），在 `T = 16,263` 下单层释放 **`508.2 MiB`**，60 层模型峰值反传显存降低 **`29.29 GiB`**。

### 2.4 TileLang 编译器与共享内存安全守卫 (`714cbf150d`, `dbabf135f9`, `a366af801a`)
- **`ReduceLowerer` 线程边界守卫 (`dbabf135f9`)**：当 kernel 包含多个消费者线程组（如 `gemm` 占 `[0, 128)`、`reduce` 占 `[32, 160)`）时，`ReduceLowerer` 解析 `thread_bounds` 并将 `run_batch` 包裹在 `thread_var in [32, 160)` 条件内，防止 `[0, 32)` 的 32 个非参与线程发生 `tid - 32` 无符号下溢并覆写归约工作区前方的 **`64` 个 float（`256 B`）** 共享内存。
- **`T.Parallel` 与 `NonRestrictCollector` (`714cbf150d`, `a366af801a`)**：通过 `tirx.convert(coalesced_width)` 支持 Python `int` 合并宽度，并豁免 `*_handle` 指针参数的非 restrict 标记以保持生成 C 代码中的 `__restrict__` 优化。

---

## 3. 确定性实验与架构师练习

- **离线验证程序**：`experiments/flashinfer_kda_plan_ssd_sm107_muse_mm_fp4_silu_bf16fp4_sm120_gdn_softmax_mhc_tilelang_phase124/run.py`
- **架构师核算练习**：给定 `vocab = 81,928`、`CTAs = 8`，验证旧整 tile 分块为 $\lceil \lceil 81928 / 2048 \rceil / 8 \rceil \times 2048 = 6 \times 2048 = 12,288$，而新 32 对齐均衡分块为 $\lceil \lceil 81928 / 8 \rceil / 32 \rceil \times 32 = 10,272$，每个最忙 CTA 减少 $2,016$ 个元素（$-16.4\%$）。
