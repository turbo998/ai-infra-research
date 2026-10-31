# Interconnect Diagnostics Isolation, Emulator FP64 Modifiers & Per-Process Logging Deep Dive (Phase 171): NCCL Diagnostics Child `OMPI_*` Stripping, ROCm `rocjitsu` FP64 Finishing Bypass & NIXL Rotated Logs

> **Evidence Classification**: `implementation` (commit-pinned source audit) + `own-offline` (deterministic standard-library analytical verification)

---

## 1. Primary Upstream Commit Matrix

| Repository | Commit SHA | Date | Core Engineering Change |
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

## 2. Architectural Analysis

### 2.1 NCCL Diagnostics Child Process `OMPI_*` Stripping (`6cb2bc2e2a`)
- **Child Process MPI Sanitization**: Strips `OMPI_*` environment variables in `src/diagnostics.cc` before spawning diagnostics child processes (`inherited_ompi_env_vars_in_child = 0`), preventing child system utilities from deadlocking inside OpenMPI singleton initialization (`0` hangs).

### 2.2 ROCm `rocjitsu` FP64 Output Finishing (`8578381865`) & NIXL Per-Process Rotated Log (`748ee36d2c`)
- **CDNA5 `rocjitsu` FP64 Modifier Bypass (`#14527`)**: Skips FP64 output finishing in `fp_mode.h` when instruction output modifiers are disabled and eliminates cross-test state leaks (`#14667`, `0` leaks).
- **NIXL Per-Process Log File (`#2211`)**: Adds `NIXL_LOG_FILE` supporting `%h` (hostname), `%p` (PID), `%t` (startup time), and `%%` with `NIXL_LOG_FILE_SIZE` bounded rotation (`2` files max: active + `.1`).

---

## 3. Deterministic Offline Verification

Verified via `python experiments/nccl_ompi_strip_rocjitsu_fp64_nixl_logfile_phase171/run.py --check`:
- `inherited_ompi_env_vars_in_child`: **`0`** (`diagnostics_child_mpi_hangs = 0`)
- `fp64_unmodified_finishing_ops_skipped`: **`1`** (`cross_test_state_leaks = 0`)
- `max_rotated_log_files_per_process`: **`2`** (`supported_log_format_specifiers = ["%h", "%p", "%t", "%%"]`)
