# Storage GDS (`cuFile` / `nvidia-fs`) & DeepSeek `3FS` USRBIO KV Cache / Checkpoint Model

This offline stdlib analytical model (`evidence_class: own-offline`) evaluates PCIe Root Complex traversal, host DRAM bounce overhead, effective throughput (`GB/s` and `GiB/s`), and P99 completion latency across five storage I/O paths for `100 GiB` KV cache reload and `1024 GiB` (`1 TiB`) distributed checkpoint flush:

1. `posix_page_cache` — POSIX VFS + Linux Page Cache (`3.0x` PCIe Root Complex passes, `32.0 GB/s` cap).
2. `o_direct_pinned_bounce` — `O_DIRECT` + pinned host bounce buffer (`2.0x` RC passes, `48.0 GB/s` cap).
3. `gds_cufile_bar1_p2p` — GPUDirect Storage (`cuFileReadAsync`/`cuFileWriteAsync` + `nvidia-fs` BAR1 P2P across local NVMe drives, `54.4 GB/s` read / `41.6 GB/s` write).
4. `hf3fs_usrbio_shm_memcpy` — `3FS` USRBIO (`/dev/shm` `hf3fs_iov` + `LMCache` `Hf3fsConnector` host `memcpy` staging, `48.0 GB/s` RC cap).
5. `hf3fs_usrbio_gpudirect_p2p` — `3FS` USRBIO + GPUDirect RDMA P2P zero-host-copy (`90.0 GB/s` CRAQ read-any / `64.8 GB/s` 3-replica CRAQ write-all).

## Usage

```bash
python experiments/storage_gds_kv_checkpoint/run.py
python experiments/storage_gds_kv_checkpoint/run.py --check
```
