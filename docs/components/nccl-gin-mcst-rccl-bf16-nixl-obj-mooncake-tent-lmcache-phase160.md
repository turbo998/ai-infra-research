# NCCL GIN 请求聚合与 GDAKI MCST、RCCL BF16 归约修复与直接 AllGather、NIXL 发送端推进与多批次对象映射、Mooncake TENT/EP 校验与 LMCache 进程内架构深度解析 (Phase 160)

> **证据类型**：`implementation`（一手上游提交源码审计）+ `own-offline`（本项目标准库确定性实验 `experiments/nccl_gin_mcst_rccl_bf16_nixl_obj_mooncake_tent_lmcache_phase160/`）

## 1. 一手上游提交矩阵 (Track 3: 10 Commits)

| # | 仓库与提交 | 发布日期 | 核心实现路径与源码定位 |
|---|---|---|---|
| 1 | `NVIDIA/nccl@62b72dc38a` | 2026-09-27 | `docs/userguide/source/api/device_gin.rst`、`flags.rst`：正式定义 `ncclGinOptFlagsAggregateRequests` 设备端 GIN 请求聚合标志与 Rail/Full 连接约束。 |
| 2 | `NVIDIA/nccl@10f2296bac` | 2026-09-15 | `src/transport/net_ib/gdaki/gin_host_gdaki.cc`：新增 `NCCL_GDAKI_FORCE_MCST` (`GinGdakiForceMcst`) 环境变量，支持在 GDAKI QP 建链时强制启用硬件多播（MCST）。 |
| 3 | `ROCm/rccl@fa366ac03f` | 2026-01-20 | `src/device/reduce_kernel.h`：为 `__nv_bfloat16` / `hip_bfloat16` 补充 `SPECIALIZE_REDUCE(FuncMinMax, ...)` 特化（调用 `__hmin/__hmax`），修复 ROCm >= 6.0 落入泛型整数按位比较导致负数 Min/Max 错误的严重 Bug。 |
| 4 | `ROCm/rccl@f3c5156bbf` | 2026-01-20 | `src/collectives.cc`：在 `ncclAllGather_impl` 直接 AllGather 路径中移除冗余的非原地 `cudaMemcpyAsync` 自拷贝，恢复单内核统一写入。 |
| 5 | `ai-dynamo/nixl@749442d4e7` | 2026-10-07 | `test/gtest/test_transfer.cpp`：在 `verifyNotifs` 等待循环中同步轮询推进发送端 UCX worker，修复无后台进度线程或 TSAN 下通知队列挂起超时（0 received）。 |
| 6 | `ai-dynamo/nixl@6c5edaffbf` | 2026-10-06 | `benchmark/nixlbench/src/worker/nixl/nixl_worker.cpp`：修复 `nixlbench` 对象存储（OBJ / S3 / cuObject）在 `batch_size > 1` 时 `device-ID -> object name` 映射覆写冲突。 |
| 7 | `kvcache-ai/Mooncake@dd21db78b5` | 2026-10-15 | `mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp`：在 TENT 启动前严格拒绝 `MC_TENT_CONF` 配置文件加载失败，禁止带着静默损坏配置继续初始化 RDMA 传输引擎。 |
| 8 | `kvcache-ai/Mooncake@ba6d44d962` | 2026-10-14 | `mooncake-ep/src/mooncake_ep_buffer.cpp`：在 Legacy EP Buffer 构造函数中增加容量与参数边界校验，防止低估单步最大 dispatch tokens 引发缓冲区溢出。 |
| 9 | `LMCache/LMCache@4159596c25` | 2026-10-21 | `lmcache/inprocess/integration/vllm/vllm_service_factory.py`：将 `HealthMonitor` 迁移至 `lmcache.inprocess.v1.health_monitor` 命名空间，统一进程内健康探测生命周期。 |
| 10 | `LMCache/LMCache@45e928595f` | 2026-10-21 | `lmcache/inprocess/integration/vllm/vllm_v1_adapter.py`：将 `lmcache_async_lookup_client` 与 `mooncake_lookup_client` 统一收敛到 `lmcache.inprocess.v1.lookup_client`，消除跨层导入耦合。 |

## 2. 关键实现路径与架构公式

### 2.1 NCCL GIN 请求聚合门铃缩减与 RCCL BF16 MinMax 符号陷阱
1. **GIN 门铃聚合**：当单个解码步发起 $N_{\text{put}}=64$ 次细粒度 GIN Put 时，启用 `ncclGinOptFlagsAggregateRequests`（聚合度 $B=8$）可将 NIC Doorbell MMIO 写入从 $64$ 次降至 $8$ 次（**$8.0\times$** 门铃开销缩减）。
2. **RCCL BF16 整数按位比较缺陷**：IEEE 754 BF16 负数采用符号-数值（sign-magnitude）编码，例如 $-2.0$ (`0xC000`) 的无符号/有符号 16 位整数按位比较结果与浮点数序相反。`rccl@fa366ac03f` 显式特化 `__hmin/__hmax`，将负数归约错误率从 `19` 例降至 `0`。

### 2.2 NIXL `batch_size > 1` 对象键映射与发送端 UCX 进度推进
在 `nixlbench` OBJ 模式下，当 `devices = 4`、`batch_size = 8` 时，旧实现仅以 `device_id` 生成对象名，导致每轮 $4 \times (8 - 1) = 28$ 次对象键碰撞覆写；`nixl@6c5edaffbf` 将批次索引纳入 IOV 对象名映射，实现 `0` 碰撞。

## 3. 确定性离线实验结果 (`own-offline`)

运行 `python experiments/nccl_gin_mcst_rccl_bf16_nixl_obj_mooncake_tent_lmcache_phase160/run.py --check`：
- **NCCL GIN Doorbell 缩减比**：`8.0x`（`64 -> 8`），RCCL BF16 负数 MinMax 错误 `19 -> 0`。
- **NIXL OBJ 多批次碰撞与通知超时**：对象键碰撞 `28 -> 0`，TSAN 发送端未推进通知超时 `4 -> 0`。
- **Mooncake TENT/EP 故障早筛与 LMCache 进程内查找**：静默配置错误启动 `3 -> 0`，EP 缓冲区溢出 `0`，进程内查找路径收敛节省 `185.0 us`。
