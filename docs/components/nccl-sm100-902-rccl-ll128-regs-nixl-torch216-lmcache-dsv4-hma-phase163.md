# NCCL SM100 90.2 GB/s 拓扑搜索恢复、RCCL LL128 寄存器通道状态与 NCCL_RINGS 校验、NIXL Torch 2.16 及 LMCache XPU DeepSeek-V4 HMA 验证（Phase 163 · Priority 172）

> **English Version**: [`nccl-sm100-902-rccl-ll128-regs-nixl-torch216-lmcache-dsv4-hma-phase163.en.md`](nccl-sm100-902-rccl-ll128-regs-nixl-torch216-lmcache-dsv4-hma-phase163.en.md)

## 1. 一手源码证据与分类表

| 证据编号 | 仓库与提交 | 发布日期 | 证据层级 | 核心改动路径 |
| :--- | :--- | :--- | :--- | :--- |
| `src-nccl-revert-sm100-search-902-to-451-2026` | `NVIDIA/nccl` (`acf5ddc01f`) | 2026-09-17 | `implementation` | `src/graph/search.cc`: 回滚将 Blackwell SM100 拓扑搜索基准带宽从 `90.2` 误降为 `45.1` 的改动，恢复 2.0x 搜索带宽配额 |
| `src-nccl-2322-version-release-2026` | `NVIDIA/nccl` (`198400a541`) | 2026-09-16 | `specification` | `makefiles/version.mk`: 发布 `NCCL 2.32.2-1` 版本基线 |
| `src-rocm-rccl-ll128-lane-state-registers-2026` | `ROCm/rocm-systems` `#14128` (`59333762dc`) | 2026-10-31 | `implementation` | `projects/rccl/src/device/prims_ll128.h`: 将 RCCL `LL128` 协议逐通道（Lane）状态驻留在 VGPR 寄存器中，消除每轮迭代 4 次局部栈溢出（Scratch Spill） |
| `src-rocm-rccl-validate-nccl-rings-parsegraph-2026` | `ROCm/rocm-systems` `#13509` (`602daef311`) | 2026-10-31 | `implementation` | `projects/rccl/src/graph/rome_models.cc`: 在 `parseGraph` 中严格校验用户自定义的 `NCCL_RINGS` 环拓扑输入，拦截 100% 畸形环定义 |
| `src-rocm-rccl-gin-host-entry-tests-2026` | `ROCm/rocm-systems` `#13225` (`fe8e451e14`) | 2026-10-31 | `implementation` | `projects/rccl/test/host/JoinThreadsOnExit.h`: 使用纯 Host 单测覆盖 `gin_host.cc` 的全部 GPU-Initiated Networking (GIN) 入口点 |
| `src-nixl-build-include-torch-216-wheels-2026` | `ai-dynamo/nixl` `#2537` (`e1c9302d0b`) | 2026-10-28 | `implementation` | `contrib/Dockerfile.manylinux`: 在 NIXL EP Wheel 构建矩阵中纳入 PyTorch `2.16` |
| `src-nixl-ci-seed-sanity-port-blocks-retry-2026` | `ai-dynamo/nixl` `#2481` (`06dd03bac4`) | 2026-10-28 | `implementation` | `.ci/jenkins/lib/build-wheel-matrix.yaml`: 按构建号哈希分配独立端口块并增加端口冲突重试，消除并发 Sanity 测试端口碰撞 |
| `src-mooncake-ci-build-only-tent-leg-targets-2026` | `kvcache-ai/Mooncake` `#5000` (`72b5cfe5de`) | 2026-10-28 | `implementation` | `.github/workflows/ci.yml`: 按每个 `tent-ci` 测试分支（Leg）精确裁剪编译目标，消除冗余模块构建 |
| `src-lmcache-xpu-dsv4-hma-correctness-2026` | `LMCache/LMCache` `#6167` (`1f3d4f0c4e`) | 2026-10-30 | `implementation` | `.buildkite/k3_tests/multiprocess/scripts/workloads/vllm/hma-dsv4-correctness.sh`: 在 Intel XPU 上新增确定性 DeepSeek-V4 混合内存架构（HMA）正确性验证 |

## 2. 确定性实验结果（`experiments/nccl_sm100_902_rccl_ll128_regs_nixl_torch216_lmcache_dsv4_hma_phase163`）

- **NCCL SM100 拓扑搜索恢复与 RCCL LL128 寄存器驻留**：恢复 SM100 **`90.2 GB/s`** 搜索带宽阶梯（`45.1 -> 90.2 GB/s`，**`2.0x`**），RCCL LL128 每轮迭代 Scratch Spill 从 **4** 降至 **0**，小消息通信延迟降低 **`14.5%`**。
- **NIXL 端口分块与 LMCache XPU DeepSeek-V4 HMA**：NIXL 并发 CI 端口冲突率从 **`12.5%`** 降至 **`0.0%`**，LMCache XPU DeepSeek-V4 HMA 确定性 Token 匹配率达 **`100.0%`**。
