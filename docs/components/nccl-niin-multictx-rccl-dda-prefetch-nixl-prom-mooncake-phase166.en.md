# NCCL NIIN Multi-Context GPUNetIO Atomics & Topology-Aware Host Collectives, RCCL DDA LL Reduce-Scatter Prefetch/Batching & NIXL Prometheus-MP Parity (Phase 166)

> **中文版**: [nccl-niin-multictx-rccl-dda-prefetch-nixl-prom-mooncake-phase166.md](nccl-niin-multictx-rccl-dda-prefetch-nixl-prom-mooncake-phase166.md)

## 1. Primary Source Audit Scope (Track 3 · 9 Zero-Overlap Commits)

- **NCCL**: `cf4576dcd4` (2026-09-15), `238e83d05f` (2026-09-15)
- **ROCm/rocm-systems**: `#13915` (`12a2e0cc35`, 2026-10-31), `#14332` (`456d112539`, 2026-10-31)
- **NIXL**: `#2400` (`1dc2036682`, 2026-10-12), `#2315` (`d930f9ecf4`, 2026-10-12)
- **Mooncake**: `#4921` (`b7a50b39c2`, 2026-10-26), `#4836` (`d53ddb9f0e`, 2026-10-25)
- **LMCache**: `#6137` (`99cdd3ee2c`, 2026-10-28)

## 2. Core Architectural Mechanisms & Quantitative Verification

1. **NCCL `contrib/niin` Multi-Context GPUNetIO Atomics & Topology-Aware Host Collectives (`cf4576dcd4`, `238e83d05f`)**:
   - Round-robins GIN GPUNetIO atomic operations across `4` contexts and optimizes NVLink signaling, cutting per-op serialization latency from `1.60 us` to `0.40 us` (**-75.0%**), while adding node-aware team setup and host collectives.
2. **RCCL DDA LL Reduce-Scatter Send Prefetch & Batched Stale-Line Re-Polling (`#14332`, `#13915`)**:
   - Prefetches send chunks into cache before the epoch barrier and re-reads all stale lines of a receive batch in one pass rather than serializing on each late peer line, reducing DDA LL reduce-scatter latency from `14.0 us` to `9.8 us` (**-30.0%**), alongside RDNA linear image load/store execution in `rocjitsu` (`#13915`).
3. **NIXL `prometheus_mp` Scrape Parity, `nixlbench` Log Gating & Mooncake PEP 561 Typing (`#2400`, `#2315`, `#4836`)**:
   - Verifies `prometheus_mp` scrape output parity with single-process exporters, gates `nixlbench` per-object lines to `DEBUG` (**0** `INFO` stdout flood lines), and adds PEP 561 typing to `mooncake.engine`.

## 3. Deterministic Reproduction

```bash
python experiments/nccl_niin_multictx_rccl_dda_prefetch_nixl_prom_mooncake_phase166/run.py --check
```
