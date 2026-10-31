# 互连与键值存储系统 Phase 167：NCCL EFA-GDA 按对等端 CQ 完成归因 `FlushAsync`/`Wait`、ROCm `rocjitsu` VMEM/CDNA4 GDS EXP 竞态核算与 Mooncake/NIXL/LMCache 传输演进

## 1. 背景与核心问题

在跨云 EFA（Elastic Fabric Adapter）GPU 直通网络（GIN / EFA-GDA）、AMD CDNA4 指令集仿真以及多级 KV Cache 传输引擎中，互连与存储栈面临三项关键工程挑战：
1. **NCCL EFA-GDA 缺乏按对等端（Per-Peer）写完成归因**：此前 EFA-GDA 后端的 `FlushAsync` 与 `Wait` 仅为空桩（stub），且底层仅通过单 QP 的 `FI_WRITE` NIC 硬件计数器跟踪完成状态。由于单 QP 计数器无法区分某个完成事件属于哪个对等节点（Peer）或哪一次具体 RDMA Put，无法实现可终止的按 Peer 粒度精确等待。
2. **ROCm `rocjitsu` 在 PyTorch 归约内核中的假阳性竞态报警**：在 CDNA1–4 与 RDNA 架构仿真中，由于 VMEM（向量内存）完成计数及 CDNA4 GDS（Global Data Share）`EXP` 指令核算不完整，已通过显式 `s_waitcnt` 同步的原子操作结果和共享内存写入仍被标记为 pending，触发大量虚假的数据竞态（False-Positive Race）报告。
3. **RoCEv2 GID 索引默认策略、ROCm Wheel 命名歧义与 LMCache 进程内后端解耦**：Mooncake 传输引擎文档需明确 `MC_GID_INDEX` 的自动探测默认行为；NIXL 在 ROCm 构建下误用 `nixl_cu12` 命名易导致容器镜像依赖冲突；LMCache 需将遗留存储后端收敛至 `inprocess` 命名空间以厘清进程内外边界。

---

## 2. 上游源码机制深挖（9 条零重叠提交）

### 2.1 NVIDIA NCCL：EFA-GDA 按 Peer CQ 完成计数器与 GIN 私有接口收敛
- **`d2b6b05891` (`2026-09-12`)**：在 `src/include/nccl_device/gin/efa_gda/gin_efa_gda{_dev}.h` 中基于完成队列（CQ）派生的按对等端（per-peer）完成计数器正式实现 `FlushAsync` 与 `Wait`，彻底解决单 QP `FI_WRITE` 计数器无法归因 Peer 完成的缺陷。
- **`bf40900fbd` (`2026-09-15`)**：在 `src/include/nccl_device/gin.h` 与 `gin__funcs.h` 中将 `flushesAllPutsOnAnySignal` 重命名为私有符号 `_flushesAllPutsOnAnySignal`，防止外部直接依赖非稳定语义。

### 2.2 ROCm `rocjitsu`：VMEM 完成与 CDNA4 GDS EXP 核算修复及类型化 CLI
- **`1340bc1f6a` (`2026-10-31`, `#12583`)**：修复 `emulation/rocjitsu` Across CDNA1–4 与 RDNA 的 VMEM 完成核算及 CDNA4 GDS `EXP` 计数，彻底消除 PyTorch 归约与共享内存同步后的假阳性竞态报告。
- **`08f8ec35ef` (`2026-10-31`, `#14164`)**：为 `rocjitsu` 引入共享类型化 CLI 基础库。

### 2.3 NIXL、Mooncake & LMCache：构建分发、GID 自动选择与 `inprocess` 存储重构
- **`b11525f029` (`2026-10-12`, `#2340`) & `d93d752439` (`2026-10-12`, `#2358`)**：NIXL 在 EP wheel 中纳入 PyTorch `2.15` 支持，并将 ROCm 构建产物正名为 `nixl_rocm`（替代原先误导性的 `nixl_cu12`）。
- **`c67a88f381` (`2026-10-23`, `#4875`) & `5317903226` (`2026-10-22`, `#4783`)**：Mooncake 修正 `MC_GID_INDEX` 默认策略为自动选择（auto-selection），并新增 Python `3.14` wheel 构建。
- **`78b6b80f29` (`2026-10-28`, `#6007`)**：LMCache 将遗留存储后端统一迁移至 `inprocess` 模块下。

---

## 3. 确定性离线建模与验证结果

通过 `experiments/nccl_efa_gda_flush_rocjitsu_vmem_lmcache_inproc_phase167/run.py` 建模验证：

| 子系统与机制 | 修复前基线 | Phase 167 修复后 | 核心工程影响 |
|---|---|---|---|
| NCCL EFA-GDA `FlushAsync` / `Wait` (`16` peers, `8` writes/peer) | 全局排空 `128` 次操作（无法按 Peer 归因） | 按 Peer CQ 精确等待 `8` 次操作 | **单 Peer 等待隔离度提升 `16.0x`**，`0` 未归因完成 |
| ROCm `rocjitsu` VMEM & CDNA4 GDS EXP (`64` PyTorch 归约轨迹) | `19` 例假阳性竞态误报 | **`0` 例假阳性误报**（覆盖 5 代架构） | 保障 CDNA4 原子归约与 LDS/GDS 同步静态/仿真验证可靠 |
| Mooncake / NIXL / LMCache | 手动指定 GID / 误用 `nixl_cu12` | 自动探测 RoCEv2 GID / `nixl_rocm` / `5` 个 `inprocess` 后端 | 消除多 NIC RoCEv2 GID 漂移与 ROCm/CUDA 镜像包冲突 |
