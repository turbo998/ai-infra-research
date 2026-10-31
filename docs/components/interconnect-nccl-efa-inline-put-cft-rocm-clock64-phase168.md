# 互连与仿真治理 Phase 168：NCCL EFA-GDA 128B WQE 内联 `PutValue` 与 2.32 CFT/NVLS 主机配置、ROCm `hip-tests` 移除固定 `clock64()` 自旋延迟及 NIXL/Mooncake/LMCache 多进程演进

## 1. 背景与核心问题

在 AWS EFA-GDA 设备发起式 RDMA、NCCL 2.32 容错/NVLS 集合通信以及 AMD ROCm 硬件/仿真测试体系中，存在三项关键系统工程问题：
1. **NCCL EFA-GDA 小载荷 `PutValue` 的双次 PCIe 事务开销与 `nccl4py` CFT/NVLS 能力盲区**：当 GPU 线程执行 8 字节 `PutValue` 时，若不将其内联编码进 128 字节 RDMA-Write WQE，NIC 需先读取 WQE 再额外发起一次 PCIe DMA 读取源缓冲区；同时 Python 控制面 `nccl4py` 亟需暴露 NCCL 2.32 的主机端 NVLS 模式（`nvls_host_mode`）及全通信器归约后的容错属性（`cft_support`、`cft_multicast_support`、`cft_counted_support`）。
2. **ROCm `hip-tests` 固定 `clock64()` 周期自旋在指令集仿真器下的 `210x` 超时膨胀**：在 `hipCGGridGroupType_old` 与 `hipGraphExecUpdate` 等测试中，通过固定 `clock64()` 周期数自旋来模拟时间延迟。由于真实 GPU 着色器时钟（约 `2100 MHz`）与仿真器有效推进速率（约 `10 MHz`）相差两个数量级，固定周期数在仿真环境下会导致测试耗时膨胀 `210x` 并引发误超时。
3. **NIXL 多进程 Prometheus 监控文档、Mooncake ROCm 硬件门禁与 LMCache 多进程模式漂移检查**：分布式 KV 传输栈需要对齐 `prometheus_mp` 共享内存指标语义、收紧夜间 Docker 构建边界并持续验证多进程存储一致性。

---

## 2. 上游源码机制深挖（8 条零重叠提交）

### 2.1 NVIDIA NCCL：EFA-GDA 128B WQE 内联 `PutValue` 与 `nccl4py` 2.32 CFT/NVLS
- **`10ca333fbb` (`2026-09-12`)**：在 `src/include/nccl_device/gin/efa_gda/gin_efa_gda.h` 中保持发送队列（SQ）投递共享，同时调用 `efa-dp-direct` 载荷编码器将 `PutValue` 直接内联写入 `128-byte` RDMA-Write WQE，将每次 `PutValue` 的 PCIe 事务数从 `2` 降至 `1`。
- **`627fa19dad` (`2026-09-12`)**：在 `bindings/nccl4py/nccl/core/` 中新增 `NCCLConfig.nvls_host_mode`（`NcclNvlsHostMode` 枚举），并在 `NCCLCommProperties` 中暴露经全通信器跨 Rank 归约后的 `cft_support`、`cft_multicast_support` 与 `cft_counted_support`。

### 2.2 ROCm `rocm-systems`：移除 `clock64()` 固定周期延迟与 `rocprof-trace-decoder` 路径修复
- **`23c4a1ed2a` (`2026-10-31`, `#14442`)**：在 `projects/hip-tests/catch/` 下移除 3 项协作组与 CUDA/HIP Graph 更新测试中的 `clock64()` 固定周期自旋等待，消除硬件与仿真器速率差异引起的超时。
- **`e57be12c61` (`2026-10-31`, `#14692`)**：修复 `rocprof-trace-decoder` 脚本安装目录位置。

### 2.3 NIXL、Mooncake & LMCache：多进程遥测、ROCm CI 门禁与多进程漂移巡检
- **`2980dc47d3` (`2026-10-12`, `#2394`) & `7791a1b71d` (`2026-10-12`, `#2247`)**：NIXL 更新 `prometheus_mp` 多进程指标存储注释与远程存储示例文档。
- **`2f0f4ca8ff` (`2026-10-20`, `#4750`)**：Mooncake 为 ROCm 硬件测试增加门禁并限制夜间 Docker 构建时长。
- **`7b41993f29` (`2026-10-28`, `#6113`)**：LMCache 完成多进程模式（multi-process mode）每日漂移检查。

---

## 3. 确定性离线建模与验证结果

通过 `experiments/nccl_efa_inline_put_cft_nvls_rocm_clock64_phase168/run.py` 验证：

| 子系统与机制 | 修复/优化前基线 | Phase 168 表现 | 核心工程影响 |
|---|---|---|---|
| NCCL EFA-GDA `PutValue` (`8B` payload in `128B` WQE) | `2` 次 PCIe 事务/Put | **`1` 次内联 WQE 事务/Put** | **PCIe 事务减半（`2.0x`）**；暴露 `3` 项全通信器 CFT 属性 |
| ROCm `hip-tests` (`clock64()` 自旋移除，`3` 项测试) | 仿真耗时膨胀 `210.0x` | **`0` 例仿真超时失败** | 解耦硬件时钟频率与功能正确性同步 |
| NIXL `prometheus_mp` / Mooncake / LMCache | 文档/构建无界 | 多进程指标对齐 + Nightly 有界构建 | 提升多进程 KV 缓存可观测性与 CI 确定性 |
