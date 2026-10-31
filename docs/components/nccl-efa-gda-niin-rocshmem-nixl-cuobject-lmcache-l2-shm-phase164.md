# NCCL EFA-GDA FlushAsync 回滚与 NIIN 非阻塞线程组、ROCm rocSHMEM 集合通信扩展、NIXL cuObject S3 加速及 LMCache XPU 长文档 L2 SHM（Phase 164 · Priority 173）

> **English Version**: [`nccl-efa-gda-niin-rocshmem-nixl-cuobject-lmcache-l2-shm-phase164.en.md`](nccl-efa-gda-niin-rocshmem-nixl-cuobject-lmcache-l2-shm-phase164.en.md)

## 1. 一手源码证据与分类表

| 证据编号 | 仓库与提交 | 发布日期 | 证据层级 | 核心改动路径 |
| :--- | :--- | :--- | :--- | :--- |
| `src-nccl-revert-efa-gda-flush-async-wait-2026` | `NVIDIA/nccl` (`78e54daaff`) | 2026-09-15 | `implementation` | `src/*`: 回滚 EFA GPU Direct Async (EFA-GDA) 中尚未闭环的 `FlushAsync` 与 `Wait` 实现，防止异步完成队列竞争 |
| `src-nccl-niin-streamline-nonblocking-threadgroup-2026` | `NVIDIA/nccl` (`ab4ab4e0bf`) | 2026-09-15 | `implementation` | `contrib/niin/*`: 精简 NIIN 非阻塞线程组（Threadgroup）传输路径，将同步屏障次数从 4 次降至 1 次（`-75.0%`） |
| `src-rocm-rocshmem-missing-collectives-apis-2026` | `ROCm/rocm-systems` `#12735` (`3ad12cc963`) | 2026-10-31 | `implementation` | `projects/rocshmem/*`: 补齐 `rocSHMEM` 集合通信缺失的设备端 API 重载（`#AIROCSHMEM-547`） |
| `src-rocm-rocprofiler-rocattach-merged-pt-load-2026` | `ROCm/rocm-systems` `#13666` (`ad34ecf876`) | 2026-10-31 | `implementation` | `projects/rocprofiler-sdk/*`: 修复 `rocattach` 在进程 ELF 内存映射出现合并 `PT_LOAD` 段时的解析失败 |
| `src-nixl-ci-cuda-cuobject-s3-acceleration-2026` | `ai-dynamo/nixl` `#2417` (`33546c2172`) | 2026-10-21 | `implementation` | `.ci/*`: 切换至预编译支持 `cuObject` S3 直通加速的 CUDA 构建镜像 |
| `src-nixl-ci-release-wheel-exact-commit-2026` | `ai-dynamo/nixl` `#2448` (`07ebb2a02c`) | 2026-10-21 | `implementation` | `.ci/*`: 强制 Release Wheel 从其构建的目标 Commit 检出执行，消除构建脚本与源码 SHA 漂移 |
| `src-mooncake-add-store-rs-tests-2026` | `kvcache-ai/Mooncake` `#4972` (`fe4f297a94`) | 2026-10-28 | `implementation` | `mooncake-store/rust/*`: 为 Mooncake Store Rust 绑定新增完整集成测试 |
| `src-mooncake-ci-tent-runtime-tone-parallel-2026` | `kvcache-ai/Mooncake` `#3600` (`b730bbae5f`) | 2026-10-27 | `implementation` | `.github/workflows/*`: 在 T-One E2E 中将 TENT 运行时作为独立并行 CI 任务开启 |
| `src-lmcache-xpu-long-doc-l2-shm-engine-metrics-2026` | `LMCache/LMCache` `#6102` (`2b7fd89a9c`) | 2026-10-28 | `implementation` | `.buildkite/k3_tests/*`: 在 Intel XPU 上启用长文档 L2 共享内存（SHM）测试与引擎驱动指标采集 |

## 2. 确定性实验结果（`experiments/nccl_efa_gda_niin_rocshmem_nixl_cuobject_lmcache_l2_shm_phase164`）

- **NCCL NIIN 非阻塞线程组与 rocSHMEM / rocattach**：NIIN 线程组同步屏障从 **4** 降至 **1**（**`-75.0%`**），`rocSHMEM` 缺失集合通信 API 与 `rocattach` 合并 `PT_LOAD` 解析错误均降至 **0**。
- **NIXL cuObject S3 与 LMCache XPU 长文档 L2 SHM**：Release Wheel 构建 Commit SHA 漂移降至 **0**，LMCache Intel XPU 长文档 L2 SHM 缓存命中与校验通过率达 **`100.0%`**。
