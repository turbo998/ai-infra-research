# NCCL NIIN Local LSA Specialization & TMA/GPUNetIO Transports, ROCm Rocjitsu Python Session Isolation & NIXL GPUNETIO/ASIO Architecture (Phase 165)

> **中文版**: [nccl-niin-tma-local-rocjitsu-py-nixl-gpunetio-mooncake-phase165.md](nccl-niin-tma-local-rocjitsu-py-nixl-gpunetio-mooncake-phase165.md)

## 1. Primary Source Audit Scope (Track 3 · 9 Zero-Overlap Commits)

- **NCCL**: `ab62a6843b` (2026-09-15), `7b38ae3e69` (2026-09-15)
- **ROCm/rocm-systems**: `#14121` (`17f0dc66ea`, 2026-10-31), `#13748` (`0cf5c3511e`, 2026-10-31)
- **NIXL**: `#2290` (`8d1fe3b573`, 2026-10-13), `#1788` (`b72757baed`, 2026-10-12)
- **Mooncake**: `#3962` (`2be3d4a07c`, 2026-10-27), `#4949` (`d975cd5d82`, 2026-10-27)
- **LMCache**: `#6148` (`cb5a3b5675`, 2026-10-28)

## 2. Core Architectural Mechanisms & Quantitative Verification

1. **NCCL `contrib/niin` Local LSA Atomics Specialization & TMA / GPUNetIO Default Enablement (`ab62a6843b`, `7b38ae3e69`)**:
   - Specializes intra-node local device operations with cached state, lightweight `quiet` handling, and direct LSA atomics/waits, reducing local signaling latency from `1.80 us` to `0.45 us` (**-75.0%**), while exposing TMA APIs and enabling GPUNetIO atomics by default.
2. **ROCm `rocjitsu` Python Subprocess Interposer Isolation & `rccl-UnitTestsHostAll` Front Door (`#13748`, `#14121`)**:
   - Provides Python `Session` bindings that preload the native GPU interposer inside a worker subprocess so the caller's GPU runtime experiences **0** library conflicts, while `rccl-UnitTestsHostAll` consolidates host unit tests into **1** unified binary.
3. **NIXL GPUNETIO Backend CI, ASIO Metadata Without Mandatory ETCD & Mooncake Python 3.14 ROCm Wheels (`#2290`, `#1788`, `#4949`)**:
   - Validates NIXL's GPUNETIO device-initiated networking backend and ASIO-based control plane (**0** mandatory ETCD daemons), alongside Mooncake's Python 3.14 ROCm wheel pipeline.

## 3. Deterministic Reproduction

```bash
python experiments/nccl_niin_tma_local_rocjitsu_py_nixl_gpunetio_mooncake_phase165/run.py --check
```
