# LMCache Device-DAX / CXL L1 Offset & Sysfs Guard, One-Shot gRPC Writer & Mooncake HA OpLog Deduplication

Deterministic Python 3.12 standard-library model verifying:
1. `LMCache/LMCache` commit `e8dc1a1daeb918aa5a1797d2ecc2c6b2cdeab1a3` (`#5874`): `get_dax_device_geometry` reads `/sys/dev/char/<major>:<minor>/{size,align}` (since `fstat` reports `0` on character devices and out-of-bounds `mmap` raises `SIGBUS` on first access), `check_dax_range` validates `2 MiB` device alignment and capacity bounds, and `(device_path, offset_in_bytes)` arena tracking rejects `5/5` invalid or symlink-aliased overlapping ranges while enabling two `512 GiB` servers/arenas to share a `1 TiB` CXL device (`0` `SIGBUS` crashes vs `2`).
2. `LMCache/LMCache` commit `dbe22ef8287c411955c37f561b5068b94c686328` (`#5906`): Lazy one-shot `_compile_response_writer` inside `compile_response_encoder_for_type` cuts structural codec compilations over `10,000` `KVEventBatch` RPCs from `10,000` to `1` (`-99.99%`, `438.0 -> 18.04 ms`, `24.28x` faster) and `0` when the handler returns a native protobuf message.
3. `kvcache-ai/Mooncake` commit `37642c5aa0fdf3e27c53ec450b24f61d729924f0` (`#4771`): Removing redundant `inc_oplog_applied_entries()` from `ApplySegmentMount`, `ApplySegmentUnmount`, and `ApplySegmentUpdate` eliminates `150` double-counted segment metrics across a `1,000`-entry HA OpLog stream (`+15.0% -> 0.0%` metric inflation).

## Run & Check
```bash
python experiments/lmcache_dax_cxl_offset_grpc_mooncake_oplog/run.py --check
```
