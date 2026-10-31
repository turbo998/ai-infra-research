# NCCL NIIN 本地 LSA 原子特化与 TMA/GPUNetIO 传输、ROCm Rocjitsu Python 会话隔离及 NIXL GPUNETIO/ASIO 去 ETCD 架构（Phase 165）

> **English Version**: [nccl-niin-tma-local-rocjitsu-py-nixl-gpunetio-mooncake-phase165.en.md](nccl-niin-tma-local-rocjitsu-py-nixl-gpunetio-mooncake-phase165.en.md)

## 1. 一手源码审计范围（Track 3 · 9 项零重叠提交）

- **NCCL**: `ab62a6843b` (2026-09-15), `7b38ae3e69` (2026-09-15)
- **ROCm/rocm-systems**: `#14121` (`17f0dc66ea`, 2026-10-31), `#13748` (`0cf5c3511e`, 2026-10-31)
- **NIXL**: `#2290` (`8d1fe3b573`, 2026-10-13), `#1788` (`b72757baed`, 2026-10-12)
- **Mooncake**: `#3962` (`2be3d4a07c`, 2026-10-27), `#4949` (`d975cd5d82`, 2026-10-27)
- **LMCache**: `#6148` (`cb5a3b5675`, 2026-10-28)

## 2. 核心架构机制与定量验证

1. **NCCL `contrib/niin` 本地 LSA 原子操作特化与 TMA / GPUNetIO 默认启用（`ab62a6843b`, `7b38ae3e69`）**：
   - 通过专用初始化、缓存本地状态、轻量级 `quiet` 处理以及直接 LSA（Load-Store Accessible）原子操作与等待指令，将节点内本地信号/同步开销从 `1.80 us` 降至 `0.45 us`（**-75.0%**），同时暴露 TMA API 并默认开启 GPUNetIO 设备端原子传输。
2. **ROCm `rocjitsu` Python 会话子进程拦截器隔离与 `rccl-UnitTestsHostAll` 统一入口（`#13748`, `#14121`）**：
   - `rocjitsu` 提供 `Session` 子进程执行机制，仅在工作子进程中预加载原生 GPU 拦截器（Interposer），使主调进程保持自身 GPU 运行时零冲突（冲突数 **0**）；同时 `rccl-UnitTestsHostAll` 将 RCCL 主机端测试收敛为单一前端二进制。
3. **NIXL GPUNETIO 后端验证、ASIO 直连移除强制 ETCD 依赖与 Mooncake Python 3.14 ROCm 构建（`#2290`, `#1788`, `#4949`）**：
   - NIXL 引入 GPUNETIO 持续集成验证并基于 ASIO 实现点对点元数据通信（强制外部 ETCD 守护进程数降为 **0**），Mooncake 则扩展支持 Python 3.14 ROCm 发行包构建。

## 3. 确定性实验复现

```bash
python experiments/nccl_niin_tma_local_rocjitsu_py_nixl_gpunetio_mooncake_phase165/run.py --check
```
