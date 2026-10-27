# TransformerEngine QB 边界无同步加载与 DistributedWeight 钩子修复、Megatron-LM MFSDP v2 执行序预取与确定性训练、TorchTitan B200 FP 融合、Mooncake XPU io_uring 暂存与 NUMA 首次触页、LMCache DAX 区间所有权、Dynamo 远端可拉取 KV 路由积分及 NIXL 追踪标识读取器（Phase 110）

> English version: [te-qb-bounds-dist-weight-megatron-mfsdp-torchtitan-mooncake-xpu-nixl-phase110.en.md](te-qb-bounds-dist-weight-megatron-mfsdp-torchtitan-mooncake-xpu-nixl-phase110.en.md) | 关联实验：`experiments/te_qb_bounds_dist_weight_megatron_mfsdp_mooncake_nixl_phase110/run.py`

## 1. 一手代码证据（2026-10-19 至 2026-10-27）

| 仓库 | Commit | 日期 | 核心机制变更 | 证据等级 |
|---|---|---|---|---|
| `TransformerEngine` | `96c0fb0e24` | 2026-10-27 | 移除 `fused_topk_with_score_function` 中 `bin_bounds` 的 `cudaMemcpyAsync + cudaStreamSynchronize` 主机同步校验，依赖设备端 `__float2int_rd` 饱和转换与 `[0, num_bins)` 钳制保证访存安全 | `implementation` |
| `TransformerEngine` | `fa4b30f09b` | 2026-10-27 | 引入 `_SavedTensorRef` 在 `saved_tensors_hooks` / 非重入激活重计算下还原 `DistributedWeight` 实例，并将 32 专家 `GroupedLinear` 的 `prepare_for_saving` CPU 开销从 `10.0 us` 降至 `5.2 us`（提速 `1.9231x`） | `implementation` + `author-benchmark` |
| `Megatron-LM` | `60547a3f41` | 2026-10-19 | MFSDP v2 用基于出现次数的 `ExecutionOrder` 替换按模块去重的 `IndexedOrder`，正确驱动重复调用模块序列（如 `[m0, m1, m0, m0, m2]`）的 All-Gather 预算预取 | `implementation` |
| `Megatron-LM` | `89b28875a8` | 2026-10-21 | Nemotron Lightning 开启确定性训练模式（`--deterministic-mode: true`、`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`、`NCCL_ALGO=Ring`、`CUBLAS_WORKSPACE_CONFIG=:4096:8` 及共享 `TRITON_CACHE_DIR`） | `implementation` |
| `torchtitan` | `c1522ec208` | 2026-10-27 | B200 CI 显式固定 `TRITON_DEFAULT_FP_FUSION="1"` 并刷新 Kimi K3 DistMuon 10 步全精度参考值（第 10 步 loss 从 `6.416478` 更新为 `6.417246`），消除 Triton 默认值变更引发的漂移 | `implementation` |
| `Mooncake` | `6b90ea5e6c` | 2026-10-27 | TENT `IOUringTransport::submitTransferTasks` 将 `MTYPE_XPU` 与 `MTYPE_CUDA` 一并纳入 `4096 B` 页对齐主机缓冲暂存，修复无 GDS 时内核无法直接 DMA XPU USM 设备地址的失败 | `implementation` |
| `Mooncake` | `c3df64c745` | 2026-10-27 | `tebench` 在无 `CAP_SYS_NICE` 容器中通过绑定目标 NUMA 节点 CPU + 逐页首次触页（first-touch）分配 DRAM，并解析 `/proc/self/numa_maps` 主导节点注册真实 NUMA MR | `implementation` |
| `LMCache` | `8afb6da466` | 2026-10-22 | DAX L2 适配器与 L1 管理器新增 `offset_gb` 与基于区间的 `owns_range(path, offset, size)` 所有权校验，支持同一台 CXL 设备按不相交区间（如 `[0, 512 GiB)` 与 `[512, 1024 GiB)`）共享给 L1/L2 或多实例 | `implementation` |
| `dynamo` | `d3fd74f489` | 2026-10-26 | KV Router 在节点选择时通过 `shared_cache_multiplier`（默认 `0.5`）对可通过 KV fetch hint 从对端拉取的 `fetchable_blocks` 计入预填充抵扣积分，同时将拉取源写入路由提示 | `implementation` |
| `nixl` | `280c3a1cd8` | 2026-10-27 | 新增无锁公开接口 `nixlAgent::getXferTraceInfo()`，读取传输句柄的 `request_id`、`execution_number`（仅统计到达后端的投递）与 W3C `traceparent` | `implementation` |

## 2. 关键实现与架构师结论

1. **消除设备到主机隐式同步与钩子类型丢失**：TE 路由器移除每次调用的 `8 B` `bin_bounds` D2H 拷贝与流同步，改用硬件饱和指令 `__float2int_rd` 加 `[0, 32)` 边界钳制确保零越界；`_SavedTensorRef` 则解决了 PyTorch `saved_tensors_hooks` 解包后丢失 `DistributedWeight` 子类身份导致反向退化为非分布式路径的隐患。
2. **CXL 区间共享与拓扑感知远端 KV 积分**：LMCache 将设备级互斥升级为 `(device_path, offset_bytes, size_bytes)` 区间互斥，使单张 `1 TiB` CXL 设备可同时承载 `512 GiB` L1 与 `512 GiB` L2；Dynamo 路由器对 `60` 个可远端拉取块给予 `0.5x` 积分抵扣，将 `100` 块原始预填充负载降至 `70.0` 等效块，使空闲可拉取节点优先于过载本地缓存节点。
