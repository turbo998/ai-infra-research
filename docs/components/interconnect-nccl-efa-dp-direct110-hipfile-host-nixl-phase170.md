# 互连数据面与主机存储后端深度剖析（Phase 170）：NCCL EFA-GDA `efa-dp-direct 1.1.0`（`backendVersion 2`）、ROCm `hipFile` Host 后端及 NIXL TSan 元数据等待

> **证据分类**：`implementation`（固定上游提交点对点源码审计）+ `own-offline`（确定性标准库解析模型验证）

---

## 1. 一手上游提交矩阵

| 仓库 | 提交 SHA | 日期 | 核心工程变更 |
|---|---|---|---|
| `NVIDIA/nccl` | `07dfce7930` | 2026-09-11 | `EFA GDA: request backendVersion 2` |
| `NVIDIA/nccl` | `6727ed4d60` | 2026-09-11 | `EFA GDA: vendor efa-dp-direct 1.1.0` |
| `ROCm/rocm-systems` | `6a167157e2` | 2026-10-30 | `feat(hipFile): Add new backend 'host' that handles read/write operations between files and registered host buffers (#10824)` |
| `ROCm/rocm-systems` | `ed0644d487` | 2026-10-30 | `fix(rocprofiler-sdk): Enable strict compiler warnings in CI builds (#14377)` |
| `ai-dynamo/nixl` | `21b6327b2e` | 2026-10-08 | `test: lengthen the md_manager remote-MD wait under sanitizers (#2349)` |
| `ai-dynamo/nixl` | `671760d4df` | 2026-10-09 | `Cleanups for OBJ S3 (#2356)` |
| `kvcache-ai/Mooncake` | `9fb270f7fc` | 2026-10-28 | `[CI/Build] Fix lint and formatting in distributed Store tests (#4982)` |
| `LMCache/LMCache` | `ad1d73891d` | 2026-10-29 | `[Docker] Include confluent-kafka in the vLLM and standalone images (#6147)` |

---

## 2. 核心架构机制解析

### 2.1 NCCL EFA-GDA `efa-dp-direct 1.1.0` 与 `backendVersion 2`（`6727ed4d60`, `07dfce7930`）
- **144B QP 布局与 128B 内联 RDMA Write**：升级内置 `amzn/efa-dp-direct` 至 `1.1.0`（`5b50aab8fa`），将 `gin_host.cc` 请求的 `backendVersion` 提升为 `2`，支持 `144-byte` QP 描述符布局、运行时 `64/128-byte` WQE 几何、`128-byte` 内联 RDMA Writes 以及 `64-bit` 请求 ID。

### 2.2 ROCm `hipFile` Host 缓冲后端（`6a167157e2`）与 NIXL TSan 元数据等待（`21b6327b2e`）
- **`hipFile` `host` 后端透明支持 CPU/GPU 缓冲（`#10824`）**：新增 `host` 后端处理文件与已注册主机内存（NIXL `DRAM_SEG`）之间的读写操作，消除 NIXL AIS 插件通过 `hipFile` 注册 CPU 缓冲时的运行时错误（`0` 错误，同时支持 GPU 与 CPU 2 类内存）。
- **NIXL ThreadSanitizer 远端元数据等待延长（`#2349`）**：将 `waitForRemoteMD` 在 Sanitizer 下的等待阈值从硬编码 `3s` 扩展至 `15s`，消除 `etcd` 失效传播超时。

---

## 3. 确定性离线验证结果

运行 `python experiments/nccl_efa_dp_direct110_hipfile_host_nixl_tsan_phase170/run.py --check`：
- `efa_gda_backend_version`: **`2`** (`efa_dp_direct_qp_bytes = 144`, `efa_inline_rdma_write_bytes = 128`, `efa_req_id_bits = 64`)
- `host_backend_dram_seg_errors`: **`0`** (`host_backend_supported_mem_types = 2`)
- `tsan_etcd_invalidation_timeouts`: **`0`** (`sanitized_tsan_md_wait_s = 15`)
