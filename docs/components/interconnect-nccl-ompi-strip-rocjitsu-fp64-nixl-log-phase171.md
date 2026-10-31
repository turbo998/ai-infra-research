# 互连诊断隔离、仿真浮点修饰与按进程日志深度剖析（Phase 171）：NCCL 诊断子进程 `OMPI_*` 剥离、ROCm `rocjitsu` FP64 跳过收尾及 NIXL 按进程轮转日志

> **证据分类**：`implementation`（固定上游提交点对点源码审计）+ `own-offline`（确定性标准库解析模型验证）

---

## 1. 一手上游提交矩阵

| 仓库 | 提交 SHA | 日期 | 核心工程变更 |
|---|---|---|---|
| `NVIDIA/nccl` | `6cb2bc2e2a` | 2026-09-11 | `Added stripping of OMPI environment from diagnostics child processes` |
| `NVIDIA/nccl` | `6e95eab9d3` | 2026-09-11 | `typo: rename 'NCCL_MAX_P2P_PEERS' to 'NCCL_P2P_MAX_PEERS' in INFO log` |
| `ROCm/rocm-systems` | `8578381865` | 2026-10-30 | `[rocjitsu] Skip FP64 output finishing when modifiers are disabled (#14527)` |
| `ROCm/rocm-systems` | `83f81615c5` | 2026-10-30 | `Fix situations where state is leaked across tests causing failures (#14667)` |
| `ai-dynamo/nixl` | `748ee36d2c` | 2026-10-07 | `logging: add optional per-process log file via NIXL_LOG_FILE (#2211)` |
| `ai-dynamo/nixl` | `733c428784` | 2026-10-07 | `chore: Update release version for NIXL (1.5.0 -> 1.6.0) (#2363)` |
| `kvcache-ai/Mooncake` | `27e0132763` | 2026-10-23 | `[Misc] Add codeowners for mooncake-conductor (#4882)` |
| `LMCache/LMCache` | `d7c0b05abd` | 2026-10-28 | `[Docker] Include pygrpc-native in the vLLM and standalone images (#6048)` |

---

## 2. 核心架构机制解析

### 2.1 NCCL 诊断子进程剥离 `OMPI_*` 环境变量（`6cb2bc2e2a`）
- **子进程 MPI 环境隔离**：在 `src/diagnostics.cc` 派生诊断子进程前主动清理继承的 `OMPI_*` 环境变量（`inherited_ompi_env_vars_in_child = 0`），防止外部系统命令误触发 OpenMPI 单例初始化挂起（`0` 次挂起），并修正 `NCCL_P2P_MAX_PEERS` 日志名称（`6e95eab9d3`）。

### 2.2 ROCm `rocjitsu` FP64 跳过收尾（`8578381865`）与 NIXL 按进程轮转日志（`748ee36d2c`）
- **CDNA5 `rocjitsu` FP64 修饰符短路（`#14527`）**：当指令未启用输出修饰符（clamp/omod）时跳过 FP64 输出收尾阶段，并修复跨测试状态泄漏（`#14667`，`0` 泄漏）。
- **NIXL 按进程文件日志与轮转（`#2211`）**：通过 `NIXL_LOG_FILE` 支持 `%h`（主机名）、`%p`（PID）、`%t`（启动时间）、`%%` 格式化占位符，并通过 `NIXL_LOG_FILE_SIZE` 限制最多保留 `2` 个文件（活动日志 + `.1` 轮转文件）。

---

## 3. 确定性离线验证结果

运行 `python experiments/nccl_ompi_strip_rocjitsu_fp64_nixl_logfile_phase171/run.py --check`：
- `inherited_ompi_env_vars_in_child`: **`0`** (`diagnostics_child_mpi_hangs = 0`)
- `fp64_unmodified_finishing_ops_skipped`: **`1`** (`cross_test_state_leaks = 0`)
- `max_rotated_log_files_per_process`: **`2`** (`supported_log_format_specifiers = ["%h", "%p", "%t", "%%"]`)
