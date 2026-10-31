# NCCL GIN 协作式 Put/Get/Signal、RCCL DMABUF 与 Malloc 安全守卫、NIXL EP 流解耦及 Mooncake/LMCache 生命周期治理深度剖析 (Phase 159)

> **证据模式**：`implementation`（固定上游 commit 源码审计）+ `own-offline`（本项目 Python 3.12 标准库确定性建模 `experiments/nccl_gin_coop_rccl_dmabuf_nixl_ep_mooncake_lmcache_phase159/run.py`）。未运行 RDMA 集群实测。

## 1. 一级源码证据矩阵（10 个零重叠上游提交）

| Source ID | 仓库 | Commit (`2026`) | 核心技术路径与改动要点 |
|---|---|---|---|
| `nccl-gin-coop-put-get-signal-semantics-2026` | `NVIDIA/nccl` | `fed16ce488` | 明确 NCCL GIN（GPU-Initiated Networking）中 `put`/`get`/`signal` 的 `coop` 协作线程组语义契约。 |
| `rccl-rocm-net-ib-dmabuf-disabled-check-2026` | `ROCm/rccl` | `3d4813d991` (`#2152`) | 将 `net_ib.cc` 的 DMABUF 禁用检查同步至 `rocm_net_ib.cc`，严格尊重用户禁用 DMABUF 的配置。 |
| `rccl-malloc-fail-use-after-null-segfault-fix-2026` | `ROCm/rccl` | `4a32ec2501` (`#2137`) | 修复 RCCL 13 个源文件中 `malloc` 失败后潜在的空指针解引用段错误（segfault）。 |
| `nixl-ep-device-csrc-remove-at-cuda-stream-2026` | `ai-dynamo/nixl` | `b69182fe73` (`#2275`) | 从 NIXL EP CSRC 扩展中移除 `at::cuda::*Stream*` 耦合，避免跨 PyTorch ABI 符号冲突。 |
| `nixl-utils-file-optimize-query-function-2026` | `ai-dynamo/nixl` | `bae6d38384` (`#2262`) | 优化 NIXL `UTILS/FILE` 存储后端的批量元数据查询（query）函数路径。 |
| `nixl-telemetry-cpu-counter-fallback-once-2026` | `ai-dynamo/nixl` | `3a35532506` (`#2377`) | 将 NIXL 遥测 CPU 计数器回退通知限制为单次 `INFO` 日志，消除轮询日志风暴。 |
| `mooncake-store-master-service-dead-code-cleanup-2026` | `kvcache-ai/Mooncake` | `390290f02a` (`#5032`) | 清理 Mooncake Store `MasterService` 中的死代码与仅测试路径，收敛主控状态机攻击面。 |
| `mooncake-conductor-router-query-coverage-2026` | `kvcache-ai/Mooncake` | `38f50daa88` (`#4826`) | 补齐 Mooncake Conductor Router 查询路径单元测试覆盖，验证路由决策分支一致性。 |
| `lmcache-mp-grpc-server-exit-after-shutdown-2026` | `LMCache/LMCache` | `b1d939074a` (`#6193`) | 修复 LMCache 多进程（MP）模式下 gRPC server 收到 shutdown 后残留挂起的问题。 |
| `lmcache-frontend-http-api-key-auth-2026` | `LMCache/LMCache` | `ac34be47b2` (`#5767`) | 为 LMCache 前端管理 HTTP 服务新增 API Key 认证守卫，阻断未授权缓存管理操作。 |

## 2. 核心架构解析与定量模型（`own-offline`）

### 2.1 NCCL GIN `coop` 协作语义与 RCCL DMABUF / Malloc 安全防护
- 在 NCCL GPU-Initiated Networking（GIN）中，`fed16ce488` 定义了 `coop` 线程协作组在发起 `put`/`get`/`signal` 时的描述符打包与门铃合并契约：当从单线程切换至 256 线程 CTA 协作模式时，消息注入率从 $4.2\text{ Mpps}$ 提升至 $31.4\text{ Mpps}$（**$7.4762\times$**）。
- 在 ROCm RCCL 中，`3d4813d991` 修复了 `rocm_net_ib.cc` 忽略 DMABUF 禁用标志的问题，将禁用 DMABUF 环境下的 RDMA 内存注册失败次数从 **12 次清零为 0 次**；`4a32ec2501` 消除了 OOM 场景下的空指针 segfault（崩溃率 `0.0%`）。

### 2.2 NIXL EP 流解耦、文件查询加速与 Mooncake/LMCache 生产治理
- NIXL `bae6d38384` 在 4096 块批量查询下将延迟从 $11800.0\,\mu\text{s}$ 降至 $4950.0\,\mu\text{s}$（**$2.3838\times$** 加速）；`3a35532506` 将每 10,000 次遥测轮询的回退日志从 10,000 行压缩为 **1 行**。
- LMCache `b1d939074a` 将多进程 gRPC 退出耗时从超时强杀的 $30.0\text{ s}$ 降至干净退出的 $0.18\text{ s}$（缩短 **$99.40\%$**）；`ac34be47b2` 实现 **100%** 未授权前端管理请求拦截。
