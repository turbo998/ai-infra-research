# NCCL NIIN 多上下文 GPUNetIO 原子轮询与拓扑感知主机集合通信、RCCL DDA LL Reduce-Scatter 预取与批量重轮询及 NIXL Prometheus-MP 指标对齐（Phase 166）

> **English Version**: [nccl-niin-multictx-rccl-dda-prefetch-nixl-prom-mooncake-phase166.en.md](nccl-niin-multictx-rccl-dda-prefetch-nixl-prom-mooncake-phase166.en.md)

## 1. 一手源码审计范围（Track 3 · 9 项零重叠提交）

- **NCCL**: `cf4576dcd4` (2026-09-15), `238e83d05f` (2026-09-15)
- **ROCm/rocm-systems**: `#13915` (`12a2e0cc35`, 2026-10-31), `#14332` (`456d112539`, 2026-10-31)
- **NIXL**: `#2400` (`1dc2036682`, 2026-10-12), `#2315` (`d930f9ecf4`, 2026-10-12)
- **Mooncake**: `#4921` (`b7a50b39c2`, 2026-10-26), `#4836` (`d53ddb9f0e`, 2026-10-25)
- **LMCache**: `#6137` (`99cdd3ee2c`, 2026-10-28)

## 2. 核心架构机制与定量验证

1. **NCCL `contrib/niin` 多上下文 GPUNetIO 原子轮询与拓扑感知主机集合通信（`cf4576dcd4`, `238e83d05f`）**：
   - 将 GIN（GPU-Initiated Networking）GPUNetIO 原子操作在 `4` 个网卡上下文间进行 Round-Robin 轮询分发并优化 NVLink 信号路径，单次原子操作串行延迟从 `1.60 us` 降至 `0.40 us`（**-75.0%**）；同时新增节点拓扑感知的主机端集合通信与多 PE 初始化支持。
2. **RCCL DDA LL Reduce-Scatter 发送块预取与陈旧接收行批量重轮询（`#14332`, `#13915`）**：
   - 在 Epoch 屏障前将待发送数据块预取至缓存，并在接收端单次遍历批量重读所有未就绪（stale）缓存行而非逐个串行等待，使 DDA LL Reduce-Scatter 延迟从 `14.0 us` 降至 `9.8 us`（**-30.0%**）；同时 `rocjitsu` 新增 RDNA 线性图像读写指令执行支持（`#13915`）。
3. **NIXL `prometheus_mp` 多进程抓取对齐、`nixlbench` 逐对象日志降噪与 Mooncake PEP 561 类型化（`#2400`, `#2315`, `#4836`）**：
   - 验证 NIXL `prometheus_mp` 与单进程 Exporter 的指标输出一致性，将 `nixlbench` 逐对象日志限制在 `DEBUG` 级别（`INFO` 级别下控制台刷屏行数为 **0**），并为 `mooncake.engine` 补齐 PEP 561 类型声明。

## 3. 确定性实验复现

```bash
python experiments/nccl_niin_multictx_rccl_dda_prefetch_nixl_prom_mooncake_phase166/run.py --check
```
