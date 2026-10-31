# NCCL 2.32.3、RCCL DMABUF 禁用门控与 amd-smi 迁移、NIXL GPUNetIO、Mooncake Loopback 元数据绑定及 LMCache gRPC 实服验证（Phase 161）

> **证据分类**：上游源码级提交分析（`implementation`：`9f99e8b8c2b5`、`12df1a11afad`、`9a443f305499`、`cd745b1f4b94`、`ca2946d0fd1c`、`f3d81669b41b`、`e174114d221c`、`acaca416ee3c`）+ 本项目确定性标准库模型验证（`own-offline`：`experiments/nccl_2323_rccl_dmabuf_amdsmi_nixl_gpunetio_mooncake_lmcache_phase161/run.py`）。

## 1. 核心架构问题与代码实现路径

1. **RCCL 尊重用户显式禁用 DMABUF（`9a443f305499`）、`rocm-smi` 切换至 `amd-smi`（`cd745b1f4b94`）与 NCCL 2.32.3-1 合入（`12df1a11afad`、`9f99e8b8c2b5`）**：
   - `9a443f305499` 修复了当用户显式设置禁用 DMABUF 时 RCCL 仍探测并启用内核 DMABUF 注册的越权路径；`cd745b1f4b94` 将设备拓扑管理从旧版 `rocm-smi` CLI 切换至原生 `amd-smi` 接口（在 `own-offline` 模型中将单次设备状态轮询耗时从 `42.0 ms` 降至 `3.5 ms`，加速 `12.0x`）。
2. **Mooncake HiCache 元数据服务 Loopback 绑定（`e174114d221c`）与 NPU `-j2` 编译并发封顶（`f3d8114d221c`/`f3d81669b41b`）**：
   - `e174114d221c` 将 HiCache Mooncake E2E 元数据服务器严格绑定到 `127.0.0.1` loopback，使暴露在通配网卡接口上的元数据端口从 `4` 降至 `0`；`f3d81669b41b` 将昇腾 NPU 编译并发限制在 `-j2`，使峰值主机内存占用从 `28.4 GiB` 降至 `9.2 GiB`。
3. **NIXL Custom-UCX GPUNetIO 子模块鉴权拉取（`ca2946d0fd1c`）与 LMCache Real gRPC Server 全模式端到端验证（`acaca416ee3c`）**：
   - 验证了 LMCache gRPC 客户端在真实 gRPC 服务端下的 `3` 种核心模式（同步、异步流式与批量查找）。

## 2. 确定性实验验证

```bash
python experiments/nccl_2323_rccl_dmabuf_amdsmi_nixl_gpunetio_mooncake_lmcache_phase161/run.py --check
```
