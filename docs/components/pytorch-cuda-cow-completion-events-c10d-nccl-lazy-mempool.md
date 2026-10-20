# PyTorch CUDA 写时复制（COW）张量显式完成事件生命周期、CUDA Graph 捕获边界硬化与 `c10d` `ProcessGroupNCCLLazy` 对称内存池转发

> **English Version**: [pytorch-cuda-cow-completion-events-c10d-nccl-lazy-mempool.en.md](pytorch-cuda-cow-completion-events-c10d-nccl-lazy-mempool.en.md)  
> **证据等级 (Evidence Class)**: `implementation`（一手上游源码核验）+ `own-offline`（`experiments/pytorch_cuda_cow_nccl_lazy_mempool/run.py` 标准库确定性复算）

---

## 1. 架构背景与核心问题

在 PyTorch 编译器（`torch.compile` / AOTInductor）、参数卸载与张量视图重塑中，`lazy_clone_storage`（`_lazy_clone`）通过写时复制（Copy-on-Write, `COW`）避免不必要的显存分配与拷贝。然而在之前的 CUDA 实现中，存在三大底层缺陷：

1. **旧版默认流同步 `cudaMemcpy` 破坏非阻塞流顺序与 CUDA Graph 重放**：`NativeCachingAllocator::copy_data` 使用同步 `cudaMemcpy` 在 legacy default stream 上执行拷贝，既无法与非阻塞计算流建立事件序，又完全缺失于 CUDA Graph 捕获图中，导致 Graph Replay 时每次都读取陈旧数据（`8/8` 次重放陈旧读取）。
2. **物化拷贝后异步旁路读悬挂与“最后引用窃取（Last-Reference Steal）”双倍显存开销**：当一个共享 COW 张量在流 $B$ 上触发写时复制（`materialize_cow`）时，仅完成流 $B$ 的拷贝并不能保证此前在其他流上提交的异步读已完成。若无显式完成追踪，运行时必须将旧引用保留在 `StorageExtraMeta::cow_retired_data` 中（导致峰值显存翻倍至 `2.0x`），或在窃取最后引用时被迫等待分配流 $A$ 上所有后续无关算子（造成 `45.0 ms` 长尾停顿）。
3. **`c10d` `ProcessGroupNCCLLazy` 缺失 `register_mem_pool` / `deregister_mem_pool` 绑定**：`nccl-lazy` 虽将 `mem_allocator` 和单边窗口转发给主通信器 `getPrimary()`，但漏绑了内存池注册/注销接口，导致 `nccl-lazy` 用户无法注册对称内存池（`symm=True`）或在释放 `MemPool` 前拆除 NVLS 窗口。

---

## 2. 一手源码实现核验 (`implementation`)

### 2.1 分配流异步物化拷贝、`cow_retired_data` 保留与 `_cow_record_done` 显式事件窃取 (`pytorch/pytorch` `20f82b619b`, `#200800`)

在 `c10/core/impl/COW.cpp`（blob `70357669379b557e585316deef6d661397f0b02b`）、`c10/core/impl/COWDeleter.cpp`（blob `bda590c8912a31b0daea6154c353daecc124043d`）、`c10/core/impl/COWDeleter.h`（blob `1b3c27c92cdd88b8519d7a3c3bd8a54ec8d287ec`）、`c10/core/impl/README-cow.md`（blob `5089906503639d62c4523f4be281c5acb7def821`）与 `c10/cuda/CUDACachingAllocator.cpp`（blob `d1e76491796982f1ac278c1dbc858fba9164f2c7`）中：

1. **分配流异步物化拷贝 (`cudaMemcpyAsync` + `waitStream`)**：
   - `NativeCachingAllocator::copy_data` 改为在 `cuda::getCurrentCUDAStream()` 上调用 `cudaMemcpyAsync`。
   - 当共享 COW 张量（`!ctx->is_unique()`）在操作流 `stream` ($B$) 上物化时，切换 `c10::StreamGuard allocation_guard(*ctx->stream())` 在原始分配/重用流 $A$ 上分配新块并入队拷贝，随后执行 `impl->waitStream(stream, *ctx->stream())` 让流 $B$ 等待流 $A$，并将旧 `DataPtr` 存入 `storage->get_extra_meta().cow_retired_data` 以保护此前未完成的异步读。
2. **`torch._C._cow_record_done(tensor)` 显式完成事件与原地窃取**：
   - `COWReference::record_done()` 在当前流记录 `COWCompletion` 事件（`done->event.record(stream)`），立即令分配重用流等待该事件（`done->event.block(reuse_stream)`），并释放 `cow_retired_data`。
   - 当兄弟引用析构或释放时，`COWDeleterContext::decrement_refcount` 将 `completion` 挂入 `completions_` 链表；若任一已离开的引用未调用 `record_done()`，则标记 `needs_stream_wait_ = true`。
   - 当最后一个剩余引用（`ctx->is_unique()`）物化时，直接原地窃取原内存块（零拷贝、零额外分配），并通过 `ctx->wait_for_completion(stream)` 仅等待 `completions_` 中记录的精确完成事件（若 `needs_stream_wait_ == false`），避免了等待分配流 $A$ 上后续无关内核。
3. **CUDA Graph 捕获边界六项铁律**：
   - `check_cow_read` 禁止在捕获内读取外部已存在的 COW 张量（`capture_id == 0 || capture_id == ref->ctx->capture_id()`）；
   - `lazy_clone_storage` 对跨流、跨捕获、私有内存池（`allocation->private_pool`）或已持有 `cow_retired_data` 的存储直接回退到 `eager_clone(storage)`；
   - `materialize_cow` 与 `record_done` 严禁跨越 `ctx->capture_id()` 边界；唯独捕获内创建的唯一输出（`is_unique()`）在捕获结束后物化时允许保持原图地址并通过 `waitStream(stream, *stream_)` 原地窃取。

### 2.2 `c10d` `ProcessGroupNCCLLazy` 对称内存池转发 (`pytorch/pytorch` `47bc8be46b`, `#201432`)

在 `torch/csrc/distributed/c10d/init.cpp`（blob `f2314ee6a8f0603b324cef670039d103cf06be8b`）与 `test/distributed/test_c10d_nccl2.py`（blob `820b9915c9b422b2324e3a091878a1b85e8fe0c6`）中：
- 为 `ProcessGroupNCCLLazy` 补齐 `register_mem_pool(pool, symm=false)` 与 `deregister_mem_pool(pool)` PyBind11 绑定，直接转发至 `self.getPrimary()->registerMemPool(pool, symm)` 与 `self.getPrimary()->deregisterMemPool(pool)`（点对点 pair communicators 不持有对称窗口注册，分配器钩子会自动将新池段同步给活跃通道）。

---

## 3. 标准库确定性离线实验 (`own-offline`)

运行 `python experiments\pytorch_cuda_cow_nccl_lazy_mempool\run.py --check` 对 `256.0 MiB` 张量双兄弟生命周期（读者在 `10.0 ms` 完成，分配流 $A$ 后续排有 `35.0 ms` 无关内核，共 `45.0 ms`）及 8 次 CUDA Graph 重放进行复算：

| COW 策略模式 | CUDA Graph 8 次重放陈旧读取 | 双兄弟物化峰值显存 (`MiB`) | `cow_retired_data` 驻留显存 | 最后引用窃取流等待停顿 (`ms`) | 额外 D2D 拷贝次数 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 旧版默认流同步 `cudaMemcpy` | `8 / 8` (100% 陈旧) | `512.0 MiB` (`2.00x`) | N/A | `45.0 ms` | `1` |
| 保守事件序 COW（未调用 `_cow_record_done`） | **`0 / 8`** | `512.0 MiB` (`2.00x`) | `256.0 MiB` | `45.0 ms` (`needs_stream_wait_=true`) | `1` |
| **显式完成事件 COW (`_cow_record_done` + 原地窃取)** | **`0 / 8`** | **`256.0 MiB` (`1.00x`, `-50.0%`)** | **`0.0 MiB`** | **`10.0 ms` (`-77.78%` 停顿削减)** | **`0` (节省 `4.0 ms`)** |

- **捕获边界矩阵验证**：6 类 CUDA Graph 边界场景中，3 类非法跨捕获访问被精确拦截（`reject_runtime_error`），其余 3 类分别按 `eager_clone`、图内事件同步与图后唯一地址原地窃取安全执行。
- **`nccl-lazy` 内存池生命周期**：在已激活 `2` 个 Ring Send/Recv 点对点通道的 4 卡通信组中，`register_mem_pool(pool, symm=True)` 将 `2` 个内存段注册至主通信器并在 `finally` 中完整注销（残留池数 `0`）。

---

## 4. 架构师设计准则与可证伪问题

1. **在编译器生成代码中插入 `_cow_record_done`**：当 TorchInductor 或自定义运行时创建了一个只读 `_lazy_clone()` 别名并在旁路流消费完毕后，必须在汇聚所有旁路流后立即调用 `torch._C._cow_record_done(tensor)`，以解锁最后引用的零拷贝原地窃取并释放 `cow_retired_data` 的双倍显存占用。
2. **CUDA Graph 捕获前物化外部 COW 输入**：切勿将已处于 COW 状态的张量直接作为 `torch.cuda.CUDAGraph` 的输入传入捕获区，必须在 `capture_begin` 前显式物化。
3. **后续可证伪问题**：当 `cudaMallocAsync`（流有序虚拟内存池）支持导出分配流与捕获 ID 元数据时，`lazy_clone_storage` 能否在不回退到 `eager_clone` 的前提下复用相同的 `COWCompletion` 事件窃取机制？
