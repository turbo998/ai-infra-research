# FlashInfer cuDNN Prefill 描述符复用、Split-K 异步代理栅栏与 TileLang NVRTC/向量化分析器隔离

> **Phase 66 · Priority 75 · Track A（GPU Kernels、编译器与量化加速）**  
> **覆盖上游源码提交**：`flashinfer-ai/flashinfer@b27f6cae7da4`、`flashinfer-ai/flashinfer@c56cd4fb7222`、`flashinfer-ai/flashinfer@c01aa7779bbb`、`flashinfer-ai/flashinfer@6b38488084e3`、`tile-ai/tilelang@35ee29507557`、`tile-ai/tilelang@fe8c34618ffb`  
> **配套确定性实验**：`experiments/flashinfer-cudnn-prefill-splitk-fence-tilelang-nvrtc/run.py` / `summary.json`

---

## 1. 架构背景与核心工程问题

在 Blackwell（`sm_100a`/`sm_103a`/`sm_107a`）与 Hopper（`H100 NVL`）高吞吐长上下文推理及混合专家（MoE）训练中，内核启动开销、异步共享内存可见性与 JIT 编译缓存一致性直接决定了端到端延迟与数值正确性：

1. **cuDNN Prefill 计划描述符（Plan Descriptor）构建开销与原位元数据突变隐患**：在 FlashInfer 的 `flashinfer/cudnn/prefill.py` 中，每次调用 `_CudnnPrefillPlan.prepare()` 若无条件执行 `_prefill_descriptor_key(**metadata.graph_kwargs(None))`，会在 Host 侧重复遍历张量 shape/stride 元组并构建完整图键；更危险的是，旧版 `__init__` 直接通过 `setattr(metadata, name, views[id(tensor)])` 原位改写调用方传入的 `metadata` 对象，若调用方复用该 `metadata` 并对 `block_tables` 执行 `as_strided_()` 原位改写，计划缓存将无法察觉 stride 布局已突变。
2. **Blackwell Dense-Projection GEMM Split-K 收集器（`_sks_sb2`）异步代理内存序竞态**：在 `flashinfer/experimental/dense_projection_gemm` 的 6 个 Split-K 双 slab 收集器内核中，生产者通过异步代理写入 `shared::cta` 并完成 mbarrier 同步后，收集器 Warp 使用通用代理（Generic Proxy）批量读取 slab 部分和。若缺少 `fence.proxy.async.shared::cta;`，Blackwell SM 硬件不保证异步代理写入对通用代理读取立即可见。
3. **CAKE MoE Grouped GEMM FP32 权重梯度（WGrad）源码克隆膨胀**：3 个 WGrad 程序此前为 `bf16` 与 `fp32` 输出维护了两套几乎完全相同的生成的 CUDA 源码（8 个程序条目），导致 JIT 库体积与维护负担翻倍。
4. **TileLang NVRTC 磁盘缓存命中丢失 `libpath` 与 `VectorizeLoop` 算术分析器状态泄漏**：TileLang 的 `NVRTCAdapter.from_database` 在从磁盘缓存加载 `.cubin` 后遗漏设置 `adapter.libpath`，导致 `@autotune` 命中缓存后保存最优配置时抛出空路径异常；同时 `src/transform/loop_vectorize.cc` 中的 `VectorizeLoop` 直接把外部 `arith::Analyzer*` 传给 `VectorizeRewriter`，导致内层循环重写时绑定的临时变量约束污染外层 Pass。

---

## 2. 第一手源码实现路径与关键修复机制

### 2.1 FlashInfer `_CudnnPrefillPlan.prepare()` 标量快路径与私有分离快照（`flashinfer@b27f6cae7da4`）

在 `flashinfer/cudnn/prefill.py` 中，`_CudnnPrefillPlan.prepare(cls, metadata, dtype, device, previous=None)` 被重构为两级校验：

- **第一级（标量与不变图属性预检）**：先检查 `previous.device == device`、`previous.dtype == dtype`、`previous.metadata.o_data_type == metadata.o_data_type`、`previous.override_enabled == enabled`、`previous.metadata._bounded_mla == metadata._bounded_mla`，并直接比对 `previous.exact_keys[True]` 中的 `(max_token_per_sequence, max_sequence_kv)`、`causal` 与 `return_lse`。
- **第二级（私有张量快照存储与形状/步长校验）**：遍历 `zip(previous.inputs, inputs, strict=True)`：
  - 若 `old is new` 或 `old.is_set_to(new)`（底层 `Storage` 与 storage offset 完全一致），保持 `same_bindings = True`；
  - 若底层指针改变但 `old.shape == new.shape` 且 `old.stride() == new.stride()`，则标记 `same_bindings = False` 但保留 `same_descriptor = True`，从而直接复用 `previous.exact_keys[True]` 免去 `_prefill_descriptor_key(...)` 重算，仅重建张量绑定；
  - 在 `__init__` 中，不再通过 `setattr(metadata, name, ...)` 污染调用方对象，而是构造 `self.inputs = tuple(None if t is None else views[id(t)] for t in inputs)` 并使用 `dataclasses.replace(metadata, ...)` 生成计划私有的 `metadata` 副本。当调用方随后执行 `metadata.block_tables.as_strided_((2, 4), (1, 2))` 时，`old.stride() != new.stride()` 立即触发完整重规划。

**作者实测性能收益（Author-Benchmark，H100 NVL，`BF16 D128, Hq/Hkv=32/8`）**：
- `B=1, Q=128, KV=8192`：Host `plan()` 耗时从 `79.12 us` 降至 `68.12 us`（**降低 `13.90%`**），`plan+run` 从 `137.36 us` 降至 `126.12 us`（**降低 `8.18%`**）。
- `B=4, Q=512, KV=512`：Host `plan()` 耗时从 `77.38 us` 降至 `65.83 us`（**降低 `14.93%`**），`plan+run` 从 `135.27 us` 降至 `122.01 us`（**降低 `9.80%`**）。

### 2.2 Split-K Slab 收集器异步代理栅栏与 247 条构建条目刷新（`flashinfer@c56cd4fb7222`）

在 `flashinfer/experimental/dense_projection_gemm` 中：
- 6 个 `_sks_sb2`（2-slab Split-K 收集器，如新增注册的 `cake_dense_projection_gemm_2307cb3cca86763a9027` 模板 `dense_proj_gemm_nn_n256_sks_sb2_pro7`，配置 `sk_slab=2, sk_sync=True`）在完成 slab 同步等待后、执行批量读取前插入 PTX `fence.proxy.async.shared::cta;`，建立 `shared::cta` 状态空间内异步代理写入到通用代理读取的严格 happens-before 关系。
- 同时刷新了 13 个 BF16 输出与 3 个 FP32 输出内核的 `st.global.L1::no_allocate.L2::evict_first.v8.b32` 融合分割存储降低（消除冗余 `+(0)` 地址偏移表达式），覆盖全部 `247` 个构建条目（`sm_100a: 80`、`sm_103a: 80`、`sm_107a: 87`）。
- 在 `dense_gate_up`、`dense_down` 与 `5120 -> 5120` BF16 WGrad 基准上，GB300 获得 **`1.008x–1.016x`** 加速，GB200 获得 **`1.003x–1.037x`** 加速。

### 2.3 CAKE MoE `WGRAD_OUT=32` 编译期特化与不可达 Softmax 裁剪（`flashinfer@c01aa7779bbb`、`flashinfer@6b38488084e3`）

- `flashinfer@c01aa7779bbb` 将 3 个 FP32 WGrad MoE Grouped GEMM 克隆源码折叠回对应的 BF16 源码，通过编译宏 `-DWGRAD_OUT=16`（BF16）与 `-DWGRAD_OUT=32`（FP32）配合 `-DWGRAD_TILE={256,512}` 动态特化输出元素位宽，`PROGRAMS` 表从 `8` 个缩减至 `5` 个，净删减 **`3,125` 行**冗余代码，且全部 `26` 个编译产物库保持 **SASS 逐字节完全一致（bitwise-identical SASS）**。
- `flashinfer@6b38488084e3` 进一步移除 `cake_blackwell_softmax` 中 3 个不可达内核，净删减 **`1,829` 行**代码（两项合计净减 **`4,954` 行**）。

### 2.4 TileLang NVRTC 缓存 `libpath` 恢复与 `VectorizeLoop` 克隆分析器（`tilelang@35ee29507557`、`tilelang@fe8c34618ffb`）

- 在 `tilelang/jit/adapter/nvrtc/adapter.py` 的 `NVRTCAdapter.from_database()` 中，调用 `adapter.lib_generator.load_lib(lib_path=kernel_lib_path)` 后补齐 `adapter.libpath = adapter.lib_generator.libpath`，修复了 NVRTC 后端命中磁盘缓存后保存 autotune 结果时的 `NoneType` 路径崩溃。
- 在 `src/transform/loop_vectorize.cc` 的 `VectorizeLoop()` 中，将 `auto rewriter = VectorizeRewriter(vectorize_hint, analyzer);` 改为先执行 `auto local_analyzer = analyzer->Clone();` 再传入 `local_analyzer.get()`，彻底隔离向量化重写器内部的临时算术约束绑定。

---

## 3. 确定性实验结果与对比矩阵

运行 `python experiments/flashinfer-cudnn-prefill-splitk-fence-tilelang-nvrtc/run.py --check` 得到以下验证指标：

| 评估维度 | 修复前基线 (Legacy) | 修复后方案 (Phase 66) | 关键量化差异 / 工程收益 |
| :--- | :--- | :--- | :--- |
| **cuDNN Prefill `plan()` (B4, Q512, KV512)** | `77.38 us` | `65.83 us` | Host 规划延迟降低 **`14.93%`**（`plan+run` 降低 **`9.80%`**） |
| **复用 `metadata` 原位 `as_strided_` 检测** | 被原位 `setattr` 掩盖（误复用旧描述符） | 私有 `detach()` 快照精准捕获 stride 变化 | `100%` 阻断错位张量绑定 |
| **Split-K Slab 收集器 (`_sks_sb2`)** | 缺少异步代理栅栏 | 插入 `fence.proxy.async.shared::cta;` | 覆盖 `247` 条目（`80/80/87`），GB200 加速达 **`1.037x`** |
| **CAKE MoE WGrad + Softmax 内核库** | `8` 个 WGrad 源码 + 3 个死 Softmax | `-DWGRAD_OUT={16,32}` 折叠为 `5` 个 | 净减 **`4,954` 行**源码，`26` 个库 SASS 零变化 |
| **TileLang NVRTC 缓存 & `VectorizeLoop`** | `libpath=None`；共享 `arith::Analyzer` | 恢复 `libpath`；`analyzer->Clone()` 隔离 | Autotune 落盘零崩溃，跨循环约束污染降至 `0` |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **计算练习**：当服务工作负载采用分段 Prefill（Chunked Prefill，每秒执行 `2,500` 次 `B=4, Q=512, KV=512` 步）且复用相同 shape 的 KV 页表视图时，`_CudnnPrefillPlan.prepare()` 每次节省 `11.55 us` CPU 规划时间，每秒可释放多少毫秒的主线程调度 CPU 预算？（答案：`2500 * 11.55 us = 28.875 ms/s`，占单核 `2.89%`）。
2. **可证伪假设**：如果在调用 `prepare(metadata, ...)` 后，调用方仅通过 `tensor.set_(new_storage, new_offset, same_shape, same_stride)` 切换底层显存块但保持 `shape` 和 `stride` 完全不变，`_CudnnPrefillPlan` 是否仍需重新调用 `_prefill_descriptor_key()`？  
   - **验证结论**：**不需要**。`same_descriptor` 保持为 `True`（复用 `previous.exact_keys[True]`），仅 `same_bindings` 变为 `False` 以重新绑定新显存地址。
