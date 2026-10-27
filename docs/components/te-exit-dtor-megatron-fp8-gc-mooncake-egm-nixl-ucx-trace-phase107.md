# TransformerEngine 进程退出安全静态缓存、Megatron-LM FP8 检查点 GC 移除、Mooncake EGM+本地缓冲解耦与 NIXL 跨进程 UCX 追踪分析 (Phase 107 · Priority 116)

> **English Version**: [te-exit-dtor-megatron-fp8-gc-mooncake-egm-nixl-ucx-trace-phase107.en.md](te-exit-dtor-megatron-fp8-gc-mooncake-egm-nixl-ucx-trace-phase107.en.md)  
> **证据边界**: `implementation` (一手源码 commits) + `own-offline` (确定性标准库建模：`experiments/te_exit_dtor_megatron_fp8_gc_mooncake_egm_nixl_ucx_trace_phase107/run.py --check`)

## 1. 一手源码证据与核心实现路径

1. **`NVIDIA/TransformerEngine@dd628d3` (`#3733`: `transformer_engine/common/fused_attn/graph_cache.h`, `util/cuda_driver.h`, `util/rtc.cpp`) & `564e8b0` (`#3692`)**：
   - **禁止在 C++ 静态析构函数中调用 CUDA 驱动/cuDNN 并开启 `MALLOC_PERTURB_=165` 验证**：在进程退出阶段（`exit()`），全局/局部静态变量的析构顺序跨编译单元未定义，且 CUDA 驱动可能已开始卸载。将 `get_graph` 的 `GraphCache` 以及 `cuda_driver::call` 的 `symbol_cache` 与 `cache_mutex` 改为堆上永生引用（`static auto &cache = *new ...`），使进程退出时的静态析构 CUDA 调用次数从 `3` 降至 **`0`**，并在 `MALLOC_PERTURB_=165`（释放内存立即覆写为 `0xA5`）下消除退出时的 use-after-destroy 段错误。
2. **`NVIDIA/Megatron-LM@29319fd` (`#8522`: `megatron/training/training.py`) & `a747bac` (`#8414`) & `pytorch/torchtitan@e3737d1` (`#5621`)**：
   - **移除 FP8 检查点保存后的冗余 `gc.collect()` 同步风暴**：此前 `save_checkpoint_and_time` 在 `if args.fp8:` 下每次保存后强制执行全量 `gc.collect()`，恰好处在各 rank 保存落盘不均衡之后与紧随其后的集合通信 `timers.log`（min/max 归约）之前。移除该阻塞调用后，在 64 卡集群（落盘偏斜 `420 ms`、全量 GC 暂停 `1,850 ms`）上将每次检查点保存后的屏障尾延迟从 `2,270.0 ms` 降至 **`420.0 ms`（每次检查点节省 `1,850.0 ms`，尾延迟降低 `5.4048x`）**。
3. **`kvcache-ai/Mooncake@829d95f` (`#4859`: `mooncake-store/src/config/egm_store_pool_config.cpp`, `mooncake-integration/store/store_py.cpp`) & `LMCache/LMCache@10dd1e6` (`#5979`)**：
   - **Mooncake Store EGM（Extended GPU Memory）池与独立 `local_buffer_size` 客户端暂存池解耦**：放宽此前启用 `enable_egm_store_pool=true` 时强制要求 `local_buffer_size=0` 的限制。EGM 池继续按 `max(slab_align, cuda_granularity)`（如 `64 MiB`）在可见 GPU 对应的 `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` 节点间对齐切分（如请求 `128.03125 GiB` 在双 NUMA 节点对齐为 **`128.0 GiB` / 每节点 `64.0 GiB`**），而独立的非零 `local_buffer_size`（如 `512 MiB`）由客户端分配器管理，使同一 Provider 既能承载 NVLink/MNNVL EGM 池，又能同时调用需要本地暂存的 `put()` / `get_buffer()` API。
4. **`ai-dynamo/nixl@13a896b` (`#2404`: `test/gtest/ucx_trace_context_test.cpp`, `mocks/trace_recorder_plugin.cpp`) & `ai-dynamo/dynamo@95f1b12` (`#16540`)**：
   - **双进程 UCX 跨 Agent 唯一 `trace_id` 传播验证与 Dynamo Operator 结构化准入默认值**：NIXL 新增独立接收端可执行程序 `ucx_trace_receiver` 与 `libtrace_backend_recorder.so` 插件，通过真实 UCX 传输验证发送端与接收端 Agent 记录的完成 span 共享同一个 `trace_id`（`16/16` 次跨进程传输追踪连续率 **`100%`**）；Dynamo Operator 重构 Kubernetes 准入 Webhook 的结构化角色默认值注入。

## 2. 架构师设计练习与可证伪问题

- **计算练习**：当 Mooncake Store 请求 `requested_egm_bytes = 128 GiB + 32 MiB`，`numa_nodes = 2`，Store slab 对齐为 `2 MiB`，CUDA 分配粒度为 `64 MiB` 时，计算公共对齐粒度 `common_align` 与双节点向下对齐后的可用 EGM 容量（`128.0 GiB`），并说明为何 `local_buffer_size = 512 MiB` 不会挤占 EGM 配额。
- **可证伪假设**：若在 C++ 动态库中使用普通函数内静态 `static std::unordered_map` 缓存 CUDA 驱动符号并在析构期被另一编译单元的静态对象访问，在 `MALLOC_PERTURB_=165` 环境变量下将触发确定性崩溃（被 `#3733` 的 `*new` 永生堆分配方案消除）。
