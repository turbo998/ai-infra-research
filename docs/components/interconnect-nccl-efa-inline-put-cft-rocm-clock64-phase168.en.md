# Interconnect & Emulation Governance Phase 168: NCCL EFA-GDA 128B WQE Inline `PutValue` & 2.32 CFT/NVLS Config, ROCm `hip-tests` `clock64()` Delay Removal, and NIXL/Mooncake/LMCache Multi-Process Hardening

## 1. Architectural Context & Core Challenges

Across AWS EFA-GDA device-side RDMA, NCCL 2.32 communicator introspection, and ROCm hardware/emulator verification, three engineering bottlenecks were resolved:
1. **Two-Step PCIe Overhead for Small `PutValue` Writes in EFA-GDA & Missing Python CFT/NVLS Introspection**: Posting 8-byte `PutValue` payloads without WQE inlining required a separate DMA read beyond the 128-byte WQE fetch, while `nccl4py` lacked bindings for NCCL 2.32 `nvls_host_mode` and communicator-wide fault-tolerance (`cft_*`) capabilities.
2. **`210x` Emulation Dilation from Fixed `clock64()` Spin Loops in ROCm `hip-tests`**: Spinning for a fixed count of `clock64()` cycles in cooperative-group and graph-exec-update tests assumed hardware shader clock rates (`~2100 MHz`), causing extreme timeout dilation under instruction-level emulation (`~10 MHz` effective).
3. **Multi-Process Telemetry & CI Governance in NIXL, Mooncake & LMCache**: Production KV fabrics needed synchronized `prometheus_mp` documentation, bounded ROCm nightly Docker builds, and verified multi-process mode drift checks.

---

## 2. Primary Upstream Mechanism Deep Dive (8 Zero-Overlap Commits)

### 2.1 NVIDIA NCCL: EFA-GDA Inline `PutValue` Encoder & `nccl4py` 2.32 CFT/NVLS Properties
- **`10ca333fbb` (`2026-09-12`)**: Encodes `PutValue` payloads directly inline into 128-byte RDMA-write WQEs via `efa-dp-direct`'s published payload encoder in `gin_efa_gda.h` while keeping SQ posting shared.
- **`627fa19dad` (`2026-09-12`)**: Exposes `NCCLConfig.nvls_host_mode` (`NcclNvlsHostMode`) and rank-reduced `cft_support`, `cft_multicast_support`, and `cft_counted_support` on `NCCLCommProperties` in `nccl4py`.

### 2.2 ROCm `rocm-systems`: `clock64()` Spin Removal & `rocprof-trace-decoder` Install Path
- **`23c4a1ed2a` (`2026-10-31`, `#14442`)**: Removes fixed `clock64()` cycle-count delays from three `hip-tests` (`hipCGGridGroupType_old.cc`, `hipGraphExecUpdate.cc`, `utils.hh`).
- **`e57be12c61` (`2026-10-31`, `#14692`)**: Fixes the `rocprof-trace-decoder` scripts installation directory.

### 2.3 NIXL, Mooncake & LMCache: `prometheus_mp` Telemetry, ROCm CI Gating & Multi-Process Mode
- **`2980dc47d3` (`2026-10-12`, `#2394`) & `7791a1b71d` (`2026-10-12`, `#2247`)**: Updates NIXL `prometheus_mp` multi-process telemetry store documentation and remote storage README assets.
- **`2f0f4ca8ff` (`2026-10-20`, `#4750`)**: Gates Mooncake ROCm hardware tests and bounds nightly Docker builds.
- **`7b41993f29` (`2026-10-28`, `#6113`)**: Verifies LMCache multi-process mode configuration against daily drift.

---

## 3. Deterministic Offline Verification Summary

Validated via `experiments/nccl_efa_inline_put_cft_nvls_rocm_clock64_phase168/run.py`:

| Subsystem | Baseline Behavior | Phase 168 Verified Behavior | Architectural Impact |
|---|---|---|---|
| NCCL EFA-GDA `PutValue` (`8B` in `128B` WQE) | `2` PCIe transactions/Put | **`1` inline WQE transaction/Put** | **`2.0x` PCIe transaction reduction**; `3` reduced CFT flags |
| ROCm `hip-tests` (`clock64()` removal across `3` tests) | `210.0x` emulator clock dilation | **`0` emulation timeout failures** | Decouples functional synchronization from shader clock rate |
| NIXL / Mooncake / LMCache Governance | Unbounded nightly / stale docs | Verified `prometheus_mp` & bounded CI | Reliable multi-process KV cache telemetry and CI SLAs |
