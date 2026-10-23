# FlashInfer Cake GDP T=1 双 Head/CTA 与 Split-V Reduce-Scatter、BF16 Cold-Chain 自调优 L2 策略、Frost 单入口 TVM-FFI 与 TileLang 步长/原子类型修复深度解析

> **语言 / Language**: [中文](flashinfer-cake-gdp-2head-cold-chain-frost-ffi.md) | [English](flashinfer-cake-gdp-2head-cold-chain-frost-ffi.en.md)
> **对应问题**: `flashinfer-cake-gdp-2head-cold-chain-frost-ffi` (`priority: 76`, Phase 67 Track A)
> **证据等级**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. 核心架构问题与背景

在 Blackwell (`sm_100a` / `sm_103a`) 推理算子与编译器栈演进中，以下四项瓶颈直接制约着单步解码 (`T=1`) 与低延迟 GEMM/MoE 的性能与正确性：

1. **Cake Gated DeltaProduct (`cake_gated_delta_product_mtp`) 单步解码的两端瓶颈**：
   - 在大 Batch / 多波次场景 (`B * HV >= 32 * sm_count`)，原有的单 Head/CTA (`l8r4`, `6d077dd3b5a74469db0d`) 每个 CTA 仅处理 1 个 `(batch, head)` 行，启动与 CTA 调度开销占比偏高。
   - 在小 Batch 场景 (`B * HV <= 160`)，`splitv8_tile8` 核函数每个子步对 `row_w` 与 `row_w+4` 执行两次独立的 5 步全 Warp `__shfl_xor_sync(16..1)` 蝶形归约（共 10 次 shuffle 指令 + 2 次 `sV` 共享内存读取），成为寄存器指令吞吐瓶颈。
2. **Blackwell 低 M (`M <= 64`) BF16 GEMM 自调优器的单次启动地板噪声**：
   - 在 B300 上，单次隔离 CUDA Graph 重放存在约 `6.38 us` 的固定启动地板（`aten::add_` 测得 `6.38 us`），且不同 Kernel 家族（Cluster Split-K > Warp Split-K > Direct）首次启动开销不同。这导致真实耗时 `2.8~6.4 us` 的候选算子在隔离单次计时中全部挤压在 `~10.24 us` 附近，令自调优器错选次优算子。
3. **Frost 低延迟 MoE 的 Python Host 组装开销**：
   - Frost 每次执行需发起 2 次 (FMA) 或 4 次 (MX TC) 底层 Kernel 启动。旧版 `@cute.jit` 入口位于单个 Kernel 粒度，每次启动都在 Python 侧重新构造 `from_dlpack` 与 `make_ptr` 描述符，导致 `T=1` 时 CPU 端耗时高达 `106~261 us`（远超 `10~27 us` 的 GPU 执行时间）。
4. **TileLang 向量化步长丢失与 CUDA 原子操作无符号类型归一化漏洞**：
   - `LoopVectorizer` 在遇到 `step != 1` 的 `For` 循环时，向 `TLVectorizer` 传递了 `(loop_var, extent, body)` 却静默丢弃了 `op->step`；同时 CUDA 代码生成器将 `AtomicMax/Min/AddRet/Store` 的指针强制转换为无符号归一化类型 `NT1`（例如 `int64 -> unsigned long long`、`bf16 -> unsigned short`），引发负数比较反转与编译崩溃。

---

## 2. 上游源码实现与关键机制分析

### 2.1 FlashInfer Cake GDP：双 Head/CTA (`l8r4x2`) 与 Split-V Reduce-Scatter (`#7128`)

在 `flashinfer-ai/flashinfer@58232f576ad772c7c7f10315073c306f9cf0a292` 中：

- **双 Head/CTA `T=1` 路径 (`cake_gdp_266d24496704b9063e28`)**：
  - 当 `T == 1`、架构为 `sm_100a`/`sm_103a` 且满足 `B * HV >= 32 * sm_count` 时，分派至 `l8r4x2` 程序。
  - 每个 CTA 打包处理 2 个连续的 `(batch, head)` 行，Grid 大小设为 `ceil(B * HV / 2) = (B * HV + 1) // 2`；当 `B * HV` 为奇数（例如 `32 * sm_count + 1` 或 `B=1713, HV=3`）时，尾 CTA 安全回退处理单个有效 Head。
- **Split-V (`splitv8_tile8`, `B * HV <= 160`) 的 Reduce-Scatter + All-Gather 归约 (`cake_gdp_f2f0c7abbf10082a6996`)**：
  - 将 Warp 32 个 lane 按半 Warp 划分所有权：`row_own = (lane_local >> 4) & 1`，使下半 Warp (`lane 0..15`) 主责 `row_a = w`、辅责 `row_b = w + 4`，上半 Warp (`lane 16..31`) 主责 `row_a = w + 4`、辅责 `row_b = w`。
  - 先通过 **1 次跨半 Warp 交换** `sum_a += __shfl_xor_sync(sum_b, 16)` 将两行的跨半部分累加至各自所有权半区，随后仅对标量 `sum_a` 执行 **4 步半 Warp 内蝶形归约** (`offset = 8, 4, 2, 1`)，每个半区仅读取 1 次 `sV[sub * 64 + row_a]` 计算 `vn_a = (v_a - sum_a) * beta`，最后通过 **1 次 All-Gather** `vn_b = __shfl_xor_sync(vn_a, 16)` 恢复搭档行。
  - 每次子步的 `__shfl_xor_sync` 数量从 **10 次降至 6 次（减少 40.0%）**，输出阶段 `o_a/o_b` 归约从 **10 次降至 5 次（减少 50.0%）**。
- **作者实测性能 (`author-benchmark`)**：
  - 在 48 组形状 (`HV in {80, 20, 10, 24} x B in {1, 16, 64, 256} x T in {1, 4, 8}, n=3`) 上：
    - **B200**: `0.98x–1.18x`（`HV=80, B=256, T=1` 从 `243.4 us` 降至 `206.1 us`，**`1.18x`**；`HV=10, B=16, T=1` 从 `5.8 us` 降至 `5.3 us`，**`1.09x`**）。
    - **GB300**: `0.99x–1.13x`（`HV=80, B=256, T=1` 从 `228.0 us` 降至 `201.7 us`，**`1.13x`**；`HV=80, B=64, T=1` 从 `62.0 us` 降至 `57.2 us`，**`1.08x`**）。

### 2.2 FlashInfer BF16 GEMM：`cold-chain:8` 自调优 L2 策略 (`#6360`)

在 `flashinfer-ai/flashinfer@aec44bda63e69b657b7fee4935a3d5b2ffe08462` 中：

- 在 `TuningConfig` 中新增 `cold_chain_launches: int = 1`，并将 `_BF16_GEMM_SM100_TUNING_CONFIG` 设为 `cold_chain_launches=8`（生成缓存标识 `l2_cache_policy="cold-chain:8"`）。
- 在一次 L2 Flush 之后，于单个被捕获的 CUDA Graph 内连续串联执行 `L=8` 次指向 `ValueProfileArena` 不同非重叠槽位的 Kernel 调用，并将总耗时除以 `8`：
  $$\hat{t}_L = \frac{t_{\text{graph}}(L)}{L} = t_{\text{steady}} + \frac{t_{\text{first\_launch\_overhead}}}{L}$$
- 对于 `M=16, N=512, K=8192` (`pdl=1`)：隔离 `L=1` 计时将最优的 `SplitK(64,8,4,12)`（稳态 `3.74 us`）测为 `12.29 us`，反而错选了 `WarpSplitK(16,8,256,16,1)`（稳态 `5.17 us`，`L=1` 测得 `10.34 us`）。启用 `cold-chain:8` 后，`SplitK` 测得 `4.03 us` < `WarpSplitK` `4.48 us`，准确命中真实最优算子。
- 同时按 `max(1, min(repeat, int(10.0 / chain_ms)))` 将单候选采样总时长封顶在 `~10 ms`（`0.01 ms -> 10` 次、`4.0 ms -> 2` 次、`50.0 ms -> 1` 次），并在 `ValueProfileArena` 中用 `_raw_storage_Span_elems(t)` 保留转置/带 Padding 张量的真实物理跨距。
- **作者实测 (`author-benchmark`, B300 60 组低 M 形状)**：Tuned vs Hand 几何平均加速比从 `1.017x`（`17/60` 单元格慢 `>5%`，最差 `0.719x`）提升至 **`1.124x`**（**`0/60`** 单元格慢 `>5%`，最差 `0.983x`）；`backend="auto"` (`pdl=1`, 30 单元格) 偏离最优超过 `5%` 的比例从 `21/30` 降至 **`1/30`**（几何平均开销从 `1.171x` 降至 `1.006x`）。

### 2.3 FlashInfer Frost 单入口 TVM-FFI (`#7127`) 与 TileLang 修复 (`#3345`, `#3307`)

1. **Frost Low-Latency MoE 单入口 TVM-FFI (`c0e51ae29b19960037ae064b0ec0c45b636a312d`)**：
   - 为 `BF16` (`bf16-ffi-v1`)、`NVFP4` (`nvfp4-ffi-v1`)、`MX FMA` (`mx-ffma-ffi-v1`) 与 `MX TC` (`mx-tc-ffi-v1`) 各编译一个完整 Tactic 粒度的 `@cute.jit` `launch(...)` TVM-FFI 函数，在编译生成的 Host 函数内一次性串联 Stage1 / Mid / Stage2（及 MX TC 的激活量化 Kernel），消除每次调用的 Python `from_dlpack` 与 `make_ptr` 开销。
   - 在 B300 (`H=2048, I=1408, E=64, top_k=6, T=1 SwiGLU`) 上，公开 API CPU 端耗时加速 **`2.85x–4.65x`**（如 `MXFP4 TC128` 从 `260.85 us` 降至 `61.85 us`，`4.22x`；预打包 Runner 从 `217.82 us` 降至 `24.90 us`，`8.75x`），而 GPU 耗时保持不变 (`10.33 us`)。
2. **TileLang 步长守卫与 CUDA 原子类型保留 (`cb26ee0fafe5`, `2573d470798a`)**：
   - `LoopVectorizer` 在向量化前显式断言 `ICHECK(!op->step.has_value() || is_one(op->step.value()))`，拒绝非单位步长循环被静默改写为步长 1。
   - `AtomicMax`、`AtomicMin`、`AtomicAddRet` 与 `AtomicStore` 直接实例化 `cuda::atomic_ref<T1, cuda::thread_scope_device>(*address)`，不再经过 `normalize_atomic_type<T1>::type` (`NT1`) 转为无符号整型，修复了有符号 `int64` 负数 `atomic_max`/`atomic_min` 大小反转（如 `-(1 << 40)` 被当成大于 `1 << 40` 的无符号数）以及 `bf16`/`fp16` `atomic_store` 编译失败问题。

---

## 3. 确定性实验验证 (`own-offline`)

运行 `python experiments/flashinfer-cake-gdp-2head-cold-chain-frost-ffi/run.py --check`：

| 验证维度 | 基准 / 修复前 | 优化 / 修复后 | 确定性结论 |
|---|---|---|---|
| Cake GDP Split-V 子步 Shuffle 数 | `10` 次 `__shfl_xor_sync` | `6` 次 `__shfl_xor_sync` | **减少 40.0%**，32-lane 数值误差 `< 1e-9` |
| Cake GDP `HV=80, B=256, T=1` (B200 / GB300) | `243.4 us` / `228.0 us` | `206.1 us` / `201.7 us` | **`1.18x` / `1.13x`** (`l8r4x2` 双 Head/CTA) |
| B300 `M=16, N=512, K=8192` 自调优选择 | `L=1` 错选 `WarpSplitK` (`5.17 us`) | `cold-chain:8` 命中 `SplitK` (`3.74 us`) | 30 格 `>5%` 失误从 `21/30` 降至 **`1/30`** |
| Frost `T=1` `MXFP4 TC128` CPU 开销 (Public / Runner) | `260.85 us` / `217.82 us` | `61.85 us` / `24.90 us` | **`4.22x` / `8.75x`**，GPU `10.33 us` 不变 |
| TileLang `int64` `atomic_max(-(1<<40), 1<<40)` | `NT1 (uint64)` 错返 `-1099511627776` | `T1 (int64)` 返 `+1099511627776` | 有符号序关系与非单位步长拦截 100% 恢复 |
