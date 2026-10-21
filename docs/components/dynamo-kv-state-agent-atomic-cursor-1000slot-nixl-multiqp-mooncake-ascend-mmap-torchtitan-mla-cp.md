# Dynamo 无锁 KV 状态代理原子游标与千槽 ZMQ 汇聚、NIXL GPUNetIO 多 QP RDMA WRITE 条带化、Mooncake 昇腾 A5 `mmap`/`memfd` 主机段及 TorchTitan 压缩潜变量 MLA 上下文并行深度解析

> **演进阶段**：Phase 45 (`priority: 54`) · **上游一数据源**：`ai-dynamo/dynamo@93ff081d4e` (`#16935`), `9906dbe5ee` (`#16568`), `5235cfc93d` (`#16818`), `ai-dynamo/nixl@bb71164c4b` (`#2215`), `kvcache-ai/Mooncake@4137ac0026` (`#4756`), `7a05855674` (`#4410`), `pytorch/torchtitan@0980e1cb1c` (`#4760`), `9698a73362` (`#5394`)  
> **确定性复现脚本**：`experiments/dynamo_kv_cursor_1000slot_nixl_multiqp_torchtitan_mla_cp/run.py --check`

---

## 1. 核心架构发现与量化结论

1. **Dynamo 无锁 `KvStateAgentStatusCell` 原子游标 (`#16935`, `93ff081d4e`)、千槽级 ZMQ 套接字池化 (`#16568`, `9906dbe5ee`) 与运行时路由策略热更新 (`#16818`, `5235cfc93d`)**：
   - 将高频更新的 `outbound_cursor` 从 `ArcSwap<KvStateAgentStatus>` 中剥离为独立的 `AtomicU64`（写路径先执行 `Release` 原子存储再切换 `ArcSwap` 快照，读路径先加载 `ArcSwap` 快照再执行 `Acquire` 原子读取并与快照保底游标取 `max`），在 `10,000` 个 chunk 入站 + `2` 次生命周期切换场景下，将遍历全线程 debt 链表的 `ArcSwap::store` 调用从 **`10,002` 次降至 `2` 次（减少 `99.98%`）**；
   - `worker_raw_topic` 将排序后的 `CacheOwnerId` 列表以 `\n` 拼接后取 SHA-256 前 **16 位十六进制字符**作为定长主题（消除 ZMQ SUB 前缀过滤下的跨 worker 误匹配与集合顺序敏感性），并通过 `--slots-per-socket=8` 将 `1,024` 个 DP 槽位的 PUB/SUB 套接字总数从 **`2,048` 个压缩至 `256` 个（减少 `8x`）**。
2. **NIXL GPUNetIO 多 QP RDMA WRITE 条带化 (`#2215`, `bb71164c4b`) 与 Mooncake 昇腾 A5 RoCE `mmap`/`memfd` 主机段 (`#4756`, `4137ac0026`) 及 CPU 到对端 GPU TCP 选择 (`#4410`, `7a05855674`)**：
   - NIXL GPUNetIO 支持 `data_qp_count = 1..4` 条带化 RDMA WRITE，采用版本化 QPN 握手头 `[0x4e585150, 2, count, count + 1]`（`count` 个 WRITE QP + `1` 个独立通知/READ QP），通过单次内核启动（每个 WRITE QP 对应 1 个 CTA）在共享准备好位置环上按 `blockIdx.x` 条带化投递，并由最后到达屏障的 CTA 在完成前置 CQ 轮询与对端先序 WRITE 票据校验后独占投递末尾 chunk（携带远程通知）；
   - Mooncake 在昇腾 A5 `roce:host` 模式下使用 2 MB THP 对齐的 `mmap`/`memfd`（`ASCEND_USE_SHMEM`）替代 `aclrtMallocHost`，通过 `SharedHostMapper` 将跨进程源地址按 `mapped_base + (dest - owner_base)` 重定址后交由 `aclrtMemcpyBatch` 批量 DMA；并在无 GPU 的 CPU TENT 客户端上无条件开启 `caps.dram_to_gpu = true`。
3. **TorchTitan Kimi-K3 / DeepSeek-V3 压缩潜变量 MLA 上下文并行 (`#4760`, `0980e1cb1c`) 与优化器内聚梯度裁剪 (`#5394`, `9698a73362`)**：
   - **All-Gather KV CP**：不再跨 CP 组聚合上投影后的完整 `k_THK`（`128 × 192 = 24,576`）与 `v_THV`（`128 × 128 = 16,384`，合计 `40,960` 元素/token），而是仅聚合归一化压缩潜变量 `kv_c_normed_TL`（`512`）与共享 RoPE 键 `k_shared_TR`（`64`，合计 **`576` 元素/token**），实现 **`71.11x` All-Gather 通信量压缩**；
   - **Ulysses CP**：先对共享 `k_shared_TR` 执行 1 次轻量 All-Gather，再将 `q_THK` 与无 RoPE 重复的 `kv_THP`（`P = 128 + 128 = 256`）打包执行 **1 次前向 token→head All-to-All**（替代原有的 Q/K/V 3 次独立 All-to-All），将全流程 All-to-All 集合通信次数从 **`4` 次降至 `2` 次**，前向 All-to-All 传输量从 `65,536` 降至 `57,344` 元素/token。

---

## 2. 跨框架核心量化指标对比

| 子系统 / 机制 | 上游提交 | 优化前指标 | 优化后指标 | 量化收益 |
|---|---|---:|---:|---:|
| Dynamo `KvStateAgentStatusCell` | `dynamo@93ff081d4e` | `10,002` `ArcSwap::store` | `2` `ArcSwap::store` | **`-99.98%` 锁债务扫描** |
| Dynamo ZMQ Socket Pooling | `dynamo@9906dbe5ee` | `2,048` sockets (`1024` slots) | `256` sockets (`slots_per_socket=8`) | **`8x` 套接字收敛** |
| NIXL GPUNetIO Multi-QP WRITE | `nixl@bb71164c4b` | 单 WRITE QP 串行投递 | `1..4` WRITE QPs 单次 Launch 条带化 | **`4x` WRITE QP 并行度** |
| TorchTitan MLA All-Gather KV CP | `torchtitan@0980e1cb1c` | `40,960` elems/tok (`k_THK + v_THV`) | `576` elems/tok (`kv_c_TL + k_TR`) | **`71.11x` 通信量下降** |
| TorchTitan MLA Ulysses CP | `torchtitan@0980e1cb1c` | `4` 次 All-to-All (`65,536` fwd elems) | `2` 次 All-to-All (`57,344` fwd elems) | **`-50%` A2A 调用次数** |
