# TransformerEngine 对称内存 All-Reduce 分桶缓存、NVFP4 `torch.compile` 元数据与 DeepGEMM Mega-MHC CUDA Graph 纪元屏障深挖

> **English Version**: [te-nvfp4-compile-symm-ar-deepgemm-mhc-epoch.en.md](te-nvfp4-compile-symm-ar-deepgemm-mhc-epoch.en.md) | **返回索引 / Back to README**: [README.md](../../README.md)

## 1. 问题定义与工程背景 (Problem Statement & Architectural Context)

在 Blackwell (`SM100`/`SM120`) 低精度训练与张量并行（TP）推理/训练路径中，内核级优化常因 **动态序列长度显存驻留膨胀**、**`torch.compile` / Dynamo 假张量（`FakeTensor`）与原生 C++ 内核分支元数据失步** 以及 **CUDA Graph 重放跨迭代硬件寄存器标识（`%gridid`）复用** 引发隐蔽的显存泄漏、编译图崩溃或陈旧部分和误读：

1. **对称内存 All-Reduce 暂存池膨胀与 CUDA Graph 悬空指针悖论（`NVIDIA/TransformerEngine` `#3709`）**：
   - 当未预分配对称张量的输入调用 `symmetric_all_reduce` 时，旧实现直接以 `tensor_numel` 作为键向 `symmetric_mem_cache` 申请对称显存。在变长序列或动态微批次（如 `rows = 1..32, cols = 256`）下，每个不同尺寸都会永久驻留一块对称显存，累计驻留元素数高达 `135,168`（最大张量 `8,192` 的 **`16.50x`**）。
   - 若简单改为“仅保留单一最大缓冲区并在扩容时释放旧缓冲区”，则先前在不同序列长度下捕获的 `torch.cuda.CUDAGraph` 会因指向已释放的对称显存地址而在 `graph.replay()` 时发生非法显存访问。
2. **NVFP4 `torch.compile` / Dynamo `FakeTensor` Swizzled Scale 与 SM120 RHT 转发失步（`NVIDIA/TransformerEngine` `#3628`, `#3669`）**：
   - NVFP4 量化内核在满足对齐条件时会直接输出 GEMM-swizzled 缩放因子（跳过独立的后置 Swizzle pass）。旧版 `BiasBwdArgs.backward_compute_fake` 在 Dynamo 追踪期间构造 `TensorSpec` 时未推导 `with_gemm_swizzled_scales`，且 `get_columnwise_shape` 对 3D/4D 输入未做 2D 展平、`_amax_columnwise` 在 `row_scaled_nvfp4=True` 时错误固定为 `(1,)`，加上 `NVFP4TensorStorage.size()` 在仅含列向数据时发出 `warnings.warn`，直接导致 `torch.compile(..., fullgraph=True)` 图断裂或メタ数据不匹配。
   - 同时，在 `SM120`/`SM121` 无 TMEM 路径的 `quantize_transpose_nvfp4.cuh` 中，`quantize_transpose` 在转发 `quantize_transpose_impl` 时将最后一个参数误传为 `nullptr` 而非 `rht_matrix`（`#3669`），导致启用列向随机 Hadamard 变换（RHT）时崩溃或结果失真。
3. **DeepGEMM `sm100_mega_mhc` 在 CUDA Graph 重放下的 `%gridid` 分裂屏障陈旧读取（`deepseek-ai/DeepGEMM` `#478`）**：
   - `sm100_mega_mhc` 持久化融合内核使用全局显存分裂屏障（`gmem_split_barriers`）同步多个 K-split 生产 CTA 与后续 Mix/Norm 消费 CTA。旧实现使用 PTX `%gridid` 编码屏障状态 `encode_state(grid_idx, num_arrived_splits)`。然而 **同一个 CUDA Graph 内核节点在每次 `cudaGraphLaunch` 重放时具有完全相同的 `%gridid`**！在第 $r+1$ 次重放开始时，若消费 CTA 早于负责 `SplitBarrier::init` 的生产 CTA 到达 `SplitBarrier::wait`，它会看到第 $r$ 次重放遗留的完成态 `encode_state(grid_idx, kNumSplits)` 并立即放行，直接读取尚未写入的陈旧分裂部分和。

---

## 2. 一手源码证据与实现机制 (First-Hand Code Evidence & Implementation Mechanics)

### 2.1 TransformerEngine 对称内存 All-Reduce 四分幂次分桶缓存 (`fd7d03b1fa`)

在 `transformer_engine/pytorch/distributed.py`（blob `e90b17e4d9`，lines `2092–2102`）中，`symmetric_all_reduce` 将请求的元素数 `tensor_numel` 向上取整至每个 2 的幂次区间内的 4 个分桶容量之一：

```python
numel = max(1, tensor_numel)
step = 1 << max(0, numel.bit_length() - 3)
capacity = -(-numel // step) * step
msg = get_symmetric_memory_tensor(capacity, tensor_dtype, tensor_device, tp_group)
msg = msg[:tensor_numel]
```

- **单次分配填充上界（$\le 25\%$）**：设 $2^{k-1} \le \text{numel} < 2^k$，则 `numel.bit_length() == k`，步长 $\text{step} = 2^{k-3} = \frac{1}{4} \cdot 2^{k-1}$。由于最坏情况下向上取整跨度不超过 $\text{step}-1$，相对填充上界严格受控于：
  $$\frac{\text{capacity} - \text{numel}}{\text{numel}} < \frac{2^{k-3}}{2^{k-1}} = 0.25 \quad (25\%)$$
  在 `rows = 1..32, cols = 256` 扫描中，最坏填充出现在 `rows = 17`（`numel = 4,352` 向上取整至 `5,120`，填充率 **`17.65%`**）。
- **累计保留空间上界与 CUDA Graph 安全共存**：在 `rows = 1..32` 扫描中，32 种输入尺寸被收敛至 16 个分桶容量（`[256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192]`），累计驻留元素数从 `135,168` 降至 **`49,152`**（最大张量的 **`6.0x`**，远低于测试断言阈值 `< 8x`，缩减 **`63.64%`**）。同时，旧分桶缓冲区在扩容至 `rows = 128`（`capacity = 32,768`）后仍保留在字典中，且 `rows = 23`（`5,888`）与 `rows = 24`（`6,144`）共享同一 `6,144` 分桶，确保已捕获的多个 CUDA Graph 在交错重放时零悬空指针。

### 2.2 NVFP4 `torch.compile` / Dynamo 元数据对齐与 SM120 RHT 转发 (`f0b1096128`, `4a3a8a8e85`)

1. **C++ 与 Dynamo 共享的 Swizzled Scale 对齐推导**：
   - `transformer_engine/pytorch/csrc/quantizer.cpp`（blob `ba0410bcac`，lines `1937–1993`）新增静态方法 `NVFP4Quantizer::gemm_swizzled_scales_alignment`，并通过 PyBind11 导出为 `tex.nvfp4_gemm_swizzled_scales_alignment`：
     - 当 `!optimize_for_gemm` 时返回 `std::nullopt`；
     - 当 `with_rht == true` 时：若 `sm_arch == 120 || sm_arch == 121`（无 TMEM，复用紧凑 1D TMA 量化器加后置 Swizzle），返回 `std::nullopt`；若 `100 <= sm_arch <= 110` 且未设置 `NVTE_NVFP4_DISABLE_RHT_CAST_FUSION`，返回对齐要求 `(for_grouped_kernel ? 128 : 64, 128)`；
     - 当 `with_rht == false` 时：仅当 `with_2d_quantization && !row_scaled_nvfp4 && !nvfp4_use_4over6 && (100 <= sm_arch <= 110)` 时返回 `(128, 128)`。
   - `transformer_engine/pytorch/dynamo/fake.py`（blob `5f856d6221`，lines `19–76`）使用 `@torch.compiler.assume_constant_result` 包装该查询，并在 `_make_fused_quantization_output_spec` 中用符号可追踪的 `rows % row_alignment == 0 and cols % col_alignment == 0` 设置 `spec.with_gemm_swizzled_scales`，使 `TensorSpec.create_metadata()` 与 C++ 真实执行完全一致。
2. **列向形状 2D 展平与 `_amax_columnwise` 维度修复**：
   - `nvfp4_tensor.py`（blob `938d9a77d4`）将 `get_columnwise_shape(shape)` 简化为 `(shape[-1], math.prod(shape[:-1]))`（例如 `(4, 16, 256)` 展平为 `(256, 64)`），并将 `row_scaled_nvfp4=True` 时的 `_amax_columnwise` 从固定 `(1,)` 修正为 `(shape[-1],)`；`nvfp4_tensor_storage.py`（blob `97b66a8341`）在 `not torch.compiler.is_compiling()` 时才发出列向 `size()` 警告。
3. **SM120/SM121 RHT 矩阵指针转发（`4a3a8a8e85`）**：
   - `transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh`（blob `f83564b35f`，line `2065`）将 `quantize_transpose_impl<...>(input, noop, output, quant_config, stream, nullptr)` 修正为转发 `rht_matrix`。

### 2.3 DeepGEMM `sm100_mega_mhc` 首行尾部 `Epoch` + `exit_count` 屏障状态机 (`5081f8513a`)

在 `deep_gemm/include/deep_gemm/layout/mega_mhc.cuh`（blob `f8a103dbac`）与 `scheduler/mega_mhc.cuh`（blob `cd9afb6fab`）中：

```cpp
static constexpr uint64_t kSplitBarrierLineBytes = 128;
static constexpr uint64_t kSplitBarrierEpochOffset = kSplitBarrierLineBytes / 2; // 64B
static constexpr uint64_t kSplitBarrierExitCountOffset = kSplitBarrierEpochOffset + sizeof(uint64_t); // 72B
```

- **零外部缓冲区扩容（复用第 0 条 128B 缓存行尾部）**：每个分裂屏障槽占 128B 缓存行，但仅前 8B 存放屏障状态字。`Epoch`（`uint64_t`，偏移 `64B`）与 `exit_count`（`uint32_t`，偏移 `72B`）恰好嵌在第 0 条屏障行的后半部分（占用 `64..76B`），无需改变 `gmem_split_barriers` 缓冲区大小或主机端 ABI。
- **无符号回绕（Unsigned Distance）防生产端超前竞态**：
  $$\text{encode\_state}(\text{epoch}, s) = (\text{epoch} + 1) \times \text{kNumSplitBarriers} \times \text{kNumMaxSplits} + s = 128(\text{epoch} + 1) + s$$
  非初始化生产 CTA 在执行 `atomic_add_rel` 到达前调用 `wait_init`：
  ```cpp
  while (ptx::ld_acq_gpu(split_barrier_ptr) - current_barrier_base >= kNumSplits) {}
  ```
  当屏障尚处于上一纪元（$\text{state} < \text{current\_barrier\_base}$）时，64 位无符号减法下溢为一个接近 $2^{64}$ 的大整数 $\ge \text{kNumSplits}$，从而阻塞生产 CTA 直至初始化 CTA 写入 `current_barrier_base`；一旦初始化完成且已有 $0 \le s < \text{kNumSplits}$ 个生产分片到达，差值 $s < \text{kNumSplits}$ 立即放行！
- **最末退场 CTA 推进纪元（`Epoch::retire`）**：在 `impls/sm100_mega_mhc.cuh`（blob `d12f8defeb`，lines `697–699`）内核末尾，释放 TMEM 后每个 CTA 的 `threadIdx.x == 0` 调用 `Epoch::retire(gmem_split_barriers, gridDim.x)`。当 `atomic_add_rel(exit_count_ptr, 1) == num_ctas - 1` 时，最末 CTA 执行 `__threadfence()`、清零 `*exit_count_ptr = 0`，并通过 `ptx::st_rel(epoch_ptr, *epoch_ptr + 1)` 推进 `epoch`，使下一轮 CUDA Graph 重放在 `cudaGridDependencySynchronize` 后读取到全新的单调递增纪元。

---

## 3. 合成离线实验与量化结论 (Deterministic Offline Experiment & Quantitative Findings)

通过标准库确定性算例 [`experiments/te_nvfp4_symm_ar_deepgemm_mhc_epoch/run.py`](../../experiments/te_nvfp4_symm_ar_deepgemm_mhc_epoch/run.py)（`own-offline`）对三项机制进行了精确复现：

| 评估维度 | 修复前基线 (Legacy Baseline) | 修复后实现 (Verified Implementation) | 架构收益 / 关键变化 |
|---|---|---|---|
| **TE 对称 All-Reduce 变长驻留 (`rows=1..32, cols=256`)** | 32 个缓冲区 / `135,168` 元素 (`16.50x` 最大张量) | 16 个分桶 / **`49,152` 元素** (**`6.00x`** 最大张量) | 驻留对称显存降低 **`63.64%`**，单次填充峰值 **`17.65%`** ($\le 25\%$) |
| **CUDA Graph 跨尺寸重放 (`rows=23,24` 后扩容 `128`)** | 单缓冲区重分配导致 **`2/2`** 捕获图指针失效 | 保留历史分桶且 `23/24` 共享 `6,144` 桶 (**`2/2` 图有效**) | 兼顾显存有界性与 CUDA Graph 地址不可变性 |
| **TE NVFP4 Dynamo `FakeTensor` 元数据对齐 (4 场景)** | `with_gemm_swizzled_scales` 失配 **`2/4`**，3D 列向未展平 | 失配 **`0/4`**，列向展平 `[256, 64]`，SM120 转发 `rht_matrix` | 消除 `torch.compile(fullgraph=True)` 编译崩溃与精度故障 |
| **DeepGEMM `sm100_mega_mhc` 8 次 CUDA Graph 重放** | `%gridid` 恒为 `42`，早到消费 CTA 发生 **`7/7` 次陈旧误读** | 首行 `64B/72B` `Epoch` 状态机，陈旧误读 **`0/7`** (`final_epoch=8`) | 零额外显存开销彻底根除 CUDA Graph 重放分裂屏障竞态 |

---

## 4. 解决方案架构师设计准则与可证伪问题 (Architect Takeaways & Falsifiable Questions)

1. **CUDA Graph 捕获下的缓存池设计铁律**：任何被捕获进 CUDA Graph 的通信/计算暂存区（如 PyTorch Symmetric Memory、NCCL 用户缓冲区）其设备虚拟地址在图生命周期内不可释放重分配；应当采用 **次幂细分分桶（如每 $2^k$ 划分 4 桶，步长 $2^{k-3}$）+ 旧桶保活** 策略，将空间开销严格限制在最大请求的常数倍以内。
2. **CUDA Graph 内核跨重放同步禁止依赖 `%gridid`**：硬件寄存器 `%gridid` 仅标识图节点实例化 ID 而非重放次数；持久化内核的全局显存屏障必须在最末 CTA 退场时通过原子计数器（`exit_count`）显式递增全局显存中的 `epoch`。
3. **后续可证伪问题**：若在 `sm100_mega_mhc` 中启用 CUDA Stream Tail Launch / Programmatic Dependent Launch (PDL)，下一轮重放的前导生产 CTA 是否可能在上一轮最末 CTA 执行 `Epoch::retire` 前启动？代码注释明确指出 `Epoch::get` 在 `cudaGridDependencySynchronize` 之后读取，因此 PDL 依赖栅栏保证了前序网格全量退场后才发布新 `epoch`。
