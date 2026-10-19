# Mooncake Transfer Engine: ConnectX-8 `Data Direct` DMA-BUF Topology & `BatchRdmaDeviceCache`

Deterministic stdlib model of `kvcache-ai/Mooncake` (`commit 1f65cf13c9f886f8db8b88b7733367b18340ebf0`):
- `mooncake-transfer-engine/include/transport/rdma_transport/data_direct_topology.h` & `.cpp`: ConnectX-8 (`CX8`) `mlx5dv_get_data_direct_sysfs_path@MLX5_1.25` + `MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT` PCI-root pairing (`firstPciRootSegment`, avoiding `75%` cross-root `ENOTSUPP` / UPI hairpin).
- `mooncake-transfer-engine/include/transport/rdma_transport/rdma_batch_cache.h` (`BatchRdmaDeviceCache`): `bufferCoversRange` O(1) snapshot cache cuts 4,096-block CPU submit overhead by `34.71x` (`1.720 -> 0.050 ms`) and speeds up 64 MiB paged KV transfer by `5.23x` vs round-robin (`4.074 -> 0.779 ms`) and `3.75x` vs uncached standard GPUDirect (`2.919 -> 0.779 ms`).
