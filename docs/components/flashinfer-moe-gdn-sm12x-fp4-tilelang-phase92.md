# Phase 92 Track 1：FlashInfer MoE 分组激活重叠、SM12x/SM100 GDN 矩阵化简、NVFP4 边界清零与 TileLang 编译器验证

## Executive Summary

本篇覆盖 `flashinfer-ai/flashinfer` 与 `tile-ai/tilelang` 的 10 项最新一手上游提交（`priority: 101`）：

1. **`flashinfer-ai/flashinfer@215f83cb27` (`#7368`)**：在 Cutlass MoE 分组 GEMM 中，Ampere 架构在 `tile_k == 128` 且满足 16B 对齐（`((ptr | (k * sizeof(T))) & 15) == 0`）时通过 4 个 warp 并发发起 128-bit `cp.async` 搬运激活（未对齐时回退标量搬运）；Hopper 架构通过 4 个 warp 并发聚合激活，且对 `BF16`/`FP16` 在聚合激活前先行发起权重 TMA 异步拷贝以实现硬件流水重叠（`FP8` 因 scale 寄存器开销保持先聚合后 TMA）。
2. **`flashinfer-ai/flashinfer@801fcd0a52` (`#6371`)**：在 SM12x GDN prefill 的 `64x64` 上三角矩阵右乘（`NewV` 与 `O2`）中，利用 `16 * k > 8 * n + 7` 静态判定跳过主对角线下方的 12 个全零 `M16xN8xK16` MMA tile，将总 MMA 指令数从 `32` 降至 `20`（精简 `37.5%`）。
3. **`flashinfer-ai/flashinfer@55b2e33697` (`#5492`)**：在 SM100/SM103 GDN prefill 的 indexed `state` 路径中计算 `state_stride_divisibility`（整除 `slot`/`head`/`V` 步长且 `<= DK=128` 的最大 2 的幂），并通过 `cute.assume(..., divby=state_stride_divisibility)` 恢复 128-bit 向量化访存，消除相比 packed `state` 高达 `3.2x` 的性能回退。
4. **`flashinfer-ai/flashinfer@f45e1d6fdd` (`#6337`)**：将 GDN decode 包装器中门控与参数张量（`a`, `b`, `A_log`, `dt_bias`）的指针对齐检查从 `16B`/`32B` 放宽至元素字节大小（`elem_bytes`），允许 `HV % 8 != 0`（如 `HV = 6`）时直接传入零拷贝 `gates.chunk(2, dim=-1)` 视图。
5. **`flashinfer-ai/flashinfer@063604bfc1` (`#7433`)**：修复 SM120 `FP16` W4A16 `_process_nvfp4_packed_scales` 对 E4M3 正次正规数（subnormal，`< 1/64`，乘 `128` 后 `< 2.0`）的截断问题，完整保留全部 `7` 个非零 E4M3 subnormal scale（`m * 2^-9, m in 1..7`）。
6. **`flashinfer-ai/flashinfer@966dcfd6b6` (`#7430`)**：在 SM120 `cake_nvfp4_attention` 的 TMA store 与 barrier 交接前插入 `fence.proxy.async.shared::cta`，确保通用代理（generic proxy）写入的 shared memory 对异步 TMA 引擎可见。
7. **`flashinfer-ai/flashinfer@b2b2bfe892` (`#7008`)**：在 `rmsnorm_fp4quant` 与 `add_rmsnorm_fp4quant` 内核的 `griddepcontrol.wait` 与 `griddepcontrol.launch_dependents` 之间，将 `128x4` swizzled scale 张量的行/列对齐补齐区（padding）显式清零，避免下游 GEMM 读取脏内存或触发额外 `memset` 内核。
8. **`flashinfer-ai/flashinfer@90ac2eeb7e` (`#7050`)**：在 cuDNN 页表（page table）组装中，当 `len(spans) == batch_size == rows` 且各请求页表宽度一致时，使用单次 `out[:, :width].copy_(indices[s0:sN].view(batch_size, width))` 替代逐行切片拷贝，同时保持 CUDA Graph 预分配缓冲区地址不变。
9. **`tile-ai/tilelang@2573d47079` (`#3307`)**：修复 `AtomicMax`/`AtomicMin` 将有符号 `int64`（`T1`）强转为无符号 `NT1`（`unsigned long long`）导致负数比较反转的缺陷，并在 `AtomicStore` 中保留目标类型（`int64`/`bf16`/`fp16`）。
10. **`tile-ai/tilelang@cb26ee0faf` (`#3345`)**：在 `VectorizeLoop` 中显式拒绝 `step != 1` 的循环（`ICHECK(!op->step.has_value() || is_one(op->step.value()))`），防止向量化器静默丢弃 `step` 并错误遍历全部 `[0, extent)` 索引。

---

## 1. Upstream Commit Analysis

### 1.1 Cutlass MoE 分组激活异步预取与 SM12x/SM100 GDN 内核优化

在 `flashinfer-ai/flashinfer@215f83cb27` (`#7368`) 中，作者重构了 Cutlass MoE grouped GEMM 的激活搬运流水：
- **Ampere (`SM80`)**：当 `tile_k == 128` 时，4 个 warp 共同参与 grouped activation 向 shared memory 的 `cp.async` 搬运。为防止非对齐指针触发硬件异常，内核先验证 `((reinterpret_cast<uintptr_t>(src_ptr) | (k * sizeof(ElementA))) & 15) == 0`，满足 16B 对齐时走 128-bit `cp.async`，否则安全回退至标量拷贝。
- **Hopper (`SM90`)**：将单 warp 激活 gather 扩展到 4 个 warp 并发执行。对于 `BF16`/`FP16` 路径，内核在 gather 激活之前先发射权重的 TMA 异步加载，使权重 HBM->SMEM 传输与激活索引 gather 完全重叠。

在 Gated Delta Rule (GDN) 方向：
- **`801fcd0a52` (`#6371`)**：SM12x GDN prefill 在计算 `NewV` 与 `O2` 时，右操作数为 `64x64` 上三角矩阵。使用 `M16xN8xK16` MMA 指令沿 `k in [0, 3]`、`n in [0, 7]` 展开共 `4 x 8 = 32` 个 tile。当 `16 * k > 8 * n + 7` 时，该 `16x8` 子块完全位于主对角线下方（全零），跳过 `k=1 (n=0..1)`、`k=2 (n=0..3)`、`k=3 (n=0..5)` 共 `2 + 4 + 6 = 12` 个 tile，消除了 `37.5%` 的冗余 Tensor Core MMA。
- **`55b2e33697` (`#5492`)**：SM100/SM103 GDN prefill 在使用 `state_indices` 索引非连续请求状态槽位时，编译器因缺少步长整除信息将状态张量退化为标量加载，导致相比紧凑 `state` 出现高达 `3.2x` 的延迟退化。通过计算 `state_stride_divisibility`（整除 `slot`/`head`/`V` 步长且不超过 `DK=128` 的最大 2 的幂）并注入 `cute.assume(..., divby=state_stride_divisibility)`，恢复了 128-bit 向量化加载。
- **`f45e1d6fdd` (`#6337`)**：GDN decode 阶段通过标量指令读取每个 head 的门控标量 `a, b, A_log, dt_bias`，并不依赖 16B/32B 向量对齐。放宽对齐检查至 `elem_bytes` 后，当 `HV = 6`（`gates` 形状为 `[B, 12]`，`chunk(2, dim=-1)` 的第二半偏移为 `12B`）时不再报错或强制拷贝。

### 1.2 SM120 NVFP4 次正规数、Proxy Fence、PDL 补齐清零与 TileLang 语义守护

- **SM120 FP16 W4A16 E4M3 Subnormal 保留 (`063604bfc1`, `#7433`)**：E4M3 格式共有 7 个正次正规数（指数位 `0000`，尾数位 `001..111`，数值为 $m \times 2^{-9} < 2^{-6} = 1/64$）。旧实现针对 BF16 路径将乘 `128` 后 `< 2.0` 的值截断，误伤了 `a_dtype == torch.float16` 下全部 7 个合法次正规缩放因子。
- **SM120 NVFP4 Attention Proxy Fence (`966dcfd6b6`, `#7430`)**：在 `cake_nvfp4_attention` 中，线程通过通用代理（generic proxy）写入 shared memory 后立即由 TMA 读取或跨 CTA 同步，必须在 barrier 交接前执行 `fence.proxy.async.shared::cta`。
- **`rmsnorm_fp4quant` 128x4 Swizzle 补齐区内核内清零 (`b2b2bfe892`, `#7008`)**：NVFP4 块缩放因子采用 `128x4` swizzled 布局，行数向上取整到 `128` 的倍数、列块数向上取整到 `4` 的倍数。当 `M=1, K=256 (block=16)` 时，有效缩放因子仅 `1 x 16 = 16` 个，而 `128x4` swizzled 缓冲区包含 `128 x 16 = 2048` 个槽位（含 `2032` 个 padding 槽位）。在 `griddepcontrol.wait` 之后、`griddepcontrol.launch_dependents` 之前由内核直接清零越界区，无需主机侧额外发起 `cudaMemsetAsync`。
- **TileLang 原子类型与向量化步长验证 (`2573d47079` `#3307`, `cb26ee0faf` `#3345`)**：TileLang 将 `AtomicMax`/`AtomicMin` 的 `cuda::atomic_ref<NT1>` 修正为 `cuda::atomic_ref<T1>`，避免有符号 `int64` 负数转无符号后大于正数；并在 `VectorizeLoop` 中对 `step != 1` 抛出编译期 `ICHECK` 错误。

---

## 2. Deterministic Simulation & Quantitative Verification

通过标准库模拟器 `experiments/flashinfer_moe_gdn_sm12x_fp4_tilelang_phase92/run.py` 生成的确定性验证结果如下：

| 模块 / 提交 | 核心指标 | 修复前基线 | 修复后实测/模拟值 | 架构收益 |
|---|---|---:|---:|---|
| `flashinfer@215f83cb27` (`#7368`) | Hopper BF16 4-warp gather + weight TMA 预取 | `680` cycles | `410` cycles | `1.6585x` 阶段加速 |
| `flashinfer@801fcd0a52` (`#6371`) | SM12x GDN `64x64` 上三角 `M16xN8xK16` MMA tiles | `32` tiles | `20` tiles | 跳过 `12` 个全零 tile (`-37.5%`) |
| `flashinfer@55b2e33697` (`#5492`) | SM100 GDN indexed `state` 步长整除因子 | `divby = 1` (`2B` 访存) | `divby = 128` (`16B` 访存) | 消除 `3.2x` 延迟退化 |
| `flashinfer@063604bfc1` (`#7433`) | SM120 FP16 W4A16 E4M3 正 subnormal 保留数 | `0 / 7` | `7 / 7` | 消除小幅值权重块归零误差 |
| `flashinfer@b2b2bfe892` (`#7008`) | `M=1, K=256` NVFP4 `128x4` swizzle 内核清零槽位 | 外部清零或脏值 | `2032` 个槽位内核内清零 | 兼容 PDL 零额外 launch |
| `flashinfer@90ac2eeb7e` (`#7050`) | `B=64` 矩形 cuDNN 页表组装拷贝次数 | `64` 次切片 copy | `1` 次 `view(64, 32)` copy | `64x` 发射精简且保 CUDA Graph 指针 |
| `tilelang@2573d47079` (`#3307`) | `[-12, 3, -5, 19, -1]` 上 `int64` `AtomicMax` / `AtomicMin` | `-1` / `3` (无符号翻转) | `19` / `-12` | 修复有符号 64 位原子归约正确性 |

---

## 3. Architectural Insights & Design Exercises

1. **上三角矩阵分块跳过的代数条件推导**：对于 `M16xN8xK16` MMA，右操作数子块覆盖行区间 $r \in [16k, 16k+15]$、列区间 $c \in [8n, 8n+7]$。上三角矩阵满足 $r > c$ 时元素全为 $0$，因此当子块的最小行号严格大于最大列号，即 $16k > 8n + 7$ 时，整个 $16 \times 8$ 右操作数子块恒为 $0$。
2. **PDL (`griddepcontrol`) 下 Padding 清零的屏障位置**：在 Hopper/Blackwell Programmatic Dependent Launch 中，一旦前驱内核调用 `griddepcontrol.launch_dependents()`，后继 GEMM 内核即可开始读取全局内存输出。因此所有 `128x4` swizzled scale padding 的清零写操作必须在 `launch_dependents()` 之前发起。

---

## 4. Verification & Falsifiable Follow-Up Questions

- **验证命令**：`python experiments/flashinfer_moe_gdn_sm12x_fp4_tilelang_phase92/run.py --check` 与 `python -m unittest tests/test_phase92_experiments.py`。
- **可证伪后续问题**：当 SM12x GDN prefill chunk 大小从 `64` 扩展到 `128` 时，采用 `M16xN8xK16` 分块跳过主对角线下方全零块的理论跳过比例是否从 $12/32 = 37.5\%$ 上升至 $56/128 = 43.75\%$？
