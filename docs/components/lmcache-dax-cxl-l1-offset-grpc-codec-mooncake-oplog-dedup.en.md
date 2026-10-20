# LMCache Device-DAX / CXL L1 Offset Partitioning & `sysfs` Pre-Mapping Guard, One-Shot gRPC Response Writer & Mooncake HA OpLog Segment Metric Deduplication

> **中文版**: [lmcache-dax-cxl-l1-offset-grpc-codec-mooncake-oplog-dedup.md](lmcache-dax-cxl-l1-offset-grpc-codec-mooncake-oplog-dedup.md)

- **Queue Question ID**: `lmcache-dax-cxl-l1-offset-grpc-codec-mooncake-oplog-dedup` (`存储`, `priority: 39`, `advanced`)
- **Evidence Class**: `implementation` (verified first-hand upstream commits) + `own-offline` (deterministic Python 3.12 standard-library verification)
- **Verified Date**: 2026-10-21 (UTC+08:00)
- **Primary Upstream Sources**:
  - `LMCache/LMCache` commit [`e8dc1a1daeb918aa5a1797d2ecc2c6b2cdeab1a3`](https://github.com/LMCache/LMCache/commit/e8dc1a1daeb918aa5a1797d2ecc2c6b2cdeab1a3) (PR `#5874`, 2026-10-20): `lmcache/utils.py` (blob `bfc3c00996`), `lmcache/v1/distributed/config.py` (blob `101f60c15e`), `lmcache/v1/distributed/memory_manager/devdax_l1_memory_manager.py` (blob `c6101a7bf3`)
  - `LMCache/LMCache` commit [`dbe22ef8287c411955c37f561b5068b94c686328`](https://github.com/LMCache/LMCache/commit/dbe22ef8287c411955c37f561b5068b94c686328) (PR `#5906`, 2026-10-20): `lmcache/v1/multiprocess/transport/grpc_impl/proto_codec.py` (blob `02100af139`)
  - `kvcache-ai/Mooncake` commit [`37642c5aa0fdf3e27c53ec450b24f61d729924f0`](https://github.com/kvcache-ai/Mooncake/commit/37642c5aa0fdf3e27c53ec450b24f61d729924f0) (PR `#4771`, 2026-10-20): `mooncake-store/src/ha/oplog/oplog_applier.cpp` (blob `f1f73baf50`)

---

## 1. Architectural Problem Statement

1. **Out-of-Bounds Device-DAX `mmap` Raising Fatal `SIGBUS` (`#5874`)**: Character devices `/dev/daxX.Y` report `fstat().st_size == 0`, and the Linux kernel accepts out-of-bounds `mmap` requests only to raise `SIGBUS` on first page fault. Furthermore, sharing a large CXL device across multiple LMCache servers requires page- and device-aligned `offset` partitioning (`--l1-devdax-offset-gb` / `offset_gb`).
2. **Per-Call gRPC Structural Response Writer Re-Compilation (`#5906`)**: `_write_response_value` in `proto_codec.py` previously re-invoked `_compile_message_codec` / `_compile_field_codec` on every non-protobuf gRPC response.
3. **Double-Counted HA Segment OpLog Metrics in Mooncake Store (`#4771`)**: `OpLogApplier::ApplySegmentMount`, `ApplySegmentUnmount`, and `ApplySegmentUpdate` incremented `oplog_applied_entries` a second time after the outer `ApplyOpLogEntry` dispatcher had already counted the entry.

---

## 2. First-Hand Code Verification & Deterministic Results (`--check`)

- **Device-DAX / CXL L1 Offset & `sysfs` Guard (`e8dc1a1dae`)**: `get_dax_device_geometry` reads `/sys/dev/char/<major>:<minor>/{size,align}` (`2 MiB` align) before `mmap`, and `check_dax_range` + `(device_path, offset_in_bytes)` arena tracking reject `5/5` invalid or symlink-aliased overlapping ranges (`0` `SIGBUS` crashes vs `2` in legacy) while cleanly partitioning a `1 TiB` (`1,024 GiB`) CXL device into two disjoint `512 GiB` arenas.
- **One-Shot gRPC Response Writer (`dbe22ef828`)**: Caching `write_value` inside `compile_response_encoder_for_type` reduces codec compilations across `10,000` `KVEventBatch` RPCs from `10,000` to **`1` (`-99.99%`, `438.0 -> 18.04 ms`, `24.28x` speedup)** and `0` when returning native protobuf messages.
- **Mooncake HA OpLog Deduplication (`37642c5aa0`)**: Removing the 3 inner `inc_oplog_applied_entries()` calls eliminates `150` double-counted segment operations across a `1,000`-entry HA stream (`1,000 / 1,000` exact).

---

## 3. Linked Evidence IDs

- `claim-lmcache-devdax-cxl-l1-offset-sysfs-guard`
- `claim-lmcache-grpc-writer-once-mooncake-oplog-dedup`
- `claim-lmcache-dax-cxl-grpc-mooncake-oplog-offline`
