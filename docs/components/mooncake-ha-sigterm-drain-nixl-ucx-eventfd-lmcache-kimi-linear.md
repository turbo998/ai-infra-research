# Mooncake Store HA Master `SIGTERM`/`SIGINT` 9 阶段有序排空与 15s 看门狗、NIXL UCX `eventfd` 控制句柄收敛与 LMCache SGLang Kimi Linear Mamba 状态零拷贝适配

> **English Version**: [mooncake-ha-sigterm-drain-nixl-ucx-eventfd-lmcache-kimi-linear.en.md](mooncake-ha-sigterm-drain-nixl-ucx-eventfd-lmcache-kimi-linear.en.md)  
> **证据等级**：`implementation`（上游固定 Commit 源码核验）+ `own-offline`（`experiments/mooncake_ha_drain_nixl_eventfd_lmcache_kimi/run.py --check`）

## 1. 一手源码定位与核心架构机制

1. **Mooncake Store HA Master `SIGTERM`/`SIGINT` 有序排空与 15s 无锁看门狗（`kvcache-ai/Mooncake` commit `3db34ceb3a02113b002b3e36ac58da8d136b306e`, `mooncake-store/include/shutdown_request.h` & `mooncake-store/src/ha/leadership/master_service_supervisor.cpp`）**：
   - 修复 HA 模式下 Master 未注册信号处理器导致 `SIGTERM` 直接中断正在执行的 RPC 与 OpLog 落盘的问题：在 `main()` 获取任何租约前，先串行启动 `StartShutdownWatchdog(kHaShutdownDeadline = 15s)` 并通过 `sigaction(..., SA_ONSTACK)`（满足 Go `c-shared` HA 后端栈需求）注册 `InstallShutdownSignalHandlers()`；
   - **信号与看门狗契约**：首个 `SIGINT`/`SIGTERM` 置位无锁粘性标志 `ShutdownRequested()` 并启动不可重置的 15 秒倒计时；若排空期间收到第二个信号，信号处理器立即调用 `_exit(128 + signo)`（对 `SIGTERM=15` 返回 `143`）；若主线程阻塞在管道写满或日志刷新超过 15 秒，看门狗线程直接调用 `_exit(EXIT_FAILURE)`；
   - **9 阶段有序排空与 `_exit(0)` 防析构解约**：Serving 阶段触发器通过 `ServingStateGate` 隔离新请求，随后严格按序停止：`RPC listener -> worker pool -> leadership monitor -> OpLog writer -> serving state -> admin server -> standby replication`，最终在看门狗保护下刷新 glog 并直接调用 `_exit(0)` 退出而不执行 C++ 栈展开（stack unwinding），防止 `coordinator` 析构函数在计划内退出路径中触发同步 etcd 租约撤销阻塞。
2. **NIXL UCX 共享进度线程 `eventfd` 替代双句柄 `pipe`（`ai-dynamo/nixl` commit `8a78b338a367d935a7b1b20385042dbb227e7c71`, `src/plugins/ucx/ucx_thread_engine.cpp`）**：
   - 将 `nixlUcxSharedThread` 线程终止控制通道从 `pipe(control_pipe)`（占用 `controlPipe_[0]` 与 `controlPipe_[1]` 共 2 个文件描述符）重构为单个 `eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)`（`controlFd_`）；
   - 每个 UCX 共享进度线程的控制 FD 消耗降低 **`50.0%`（`2 -> 1`）**，在 256 个进度线程规模下直接节省 `256` 个进程 FD，且 `EFD_NONBLOCK` 64 位计数器写入天然合并多次停止通知并避免管道满阻塞。
3. **LMCache SGLang Kimi Linear Mamba 状态 `slice_axis=1` 零拷贝适配（`LMCache/LMCache` commit `e7a4f2380bc3717343ca06519a69187aa36745f9`, `lmcache/integration/sglang/unified_kv_adapter.py`）**：
   - 在 `SGLangUnifiedKVAdapter._resolve_mamba_pool_tensors` 中移除 `if slice_axis != 0: raise NotImplementedError(...)` 的误拦截：SGLang `mamba_pool` 始终以 `[slot, ...]` 作为首维暴露传输状态张量，`slice_axis`（如 Kimi Linear `conv` 状态的 `slice_axis=1`）仅描述单个 slot 内部的 TP 切分维度，不影响同 TP 规模下按完整 slot 进行的存储/恢复布局；
   - 保留 `if not layer_tensor.is_contiguous(): raise NotImplementedError(...)` 连续内存契约，确保线缆视图（如 `(5, 3, 7) -> (5, 1, 1, 21)`）维持零拷贝 `data_ptr()` 一致性。

## 2. 确定性离线实验核验（`own-offline`）

运行 `python experiments\mooncake_ha_drain_nixl_eventfd_lmcache_kimi\run.py --check`：
- **Mooncake HA SIGTERM 排空**：完整验证 `9/9` 阶段有序排空（未刷盘 OpLog 丢失从旧版 `3` 批降至 **`0`**），15s 超时看门狗准确触发 `_exit(1)`，二次 `SIGTERM` 立即返回退出码 **`143`（`128 + 15`）**；
- **NIXL UCX `eventfd`**：控制 FD 消耗减少 **`50.0%`**（256 线程节省 **`256`** 个 FD），3 次并发唤醒信号无阻塞合并为单次 64 位计数读取；
- **LMCache Kimi Linear Mamba 适配**：可接受 Mamba 状态从 `1/2` 提升至 **`2/2`**（同时支持 `ssm` `slice_axis=0` 与 Kimi Linear `conv` `slice_axis=1`），非连续张量 `1/1` 被拦截，零拷贝视图形状 `[5, 1, 1, 21]` 验证通过。

## 3. 架构师设计练习与可证伪问题

- **练习**：为何 Mooncake Store HA Master 在 `InstallShutdownSignalHandlers()` 中必须指定 `SA_ONSTACK`？
  - **解答**：Master 进程内集成了 Go `c-shared` HA 后端（如 etcd 客户端），Go 协程运行在较小的独立栈上；若信号投递到正在运行 Go 代码的线程且未启用 `SA_ONSTACK` 信号专用栈（`sigaltstack`），C 信号处理器会直接击穿 Go 协程栈导致进程崩溃。
- **可证伪问题**：若跨不同 TP 度（例如 Prefill `TP=4` 到 Decode `TP=8`）传输 Kimi Linear `slice_axis=1` 的 `conv` 状态，为何不能直接复用同 TP 的整块 `[slot, 1, 1, 21]` 字节视图？
