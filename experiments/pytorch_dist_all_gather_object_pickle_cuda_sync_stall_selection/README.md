# PyTorch `dist.all_gather_object` & `broadcast_object_list` Implicit CUDA Sync Stall vs. Fixed-Schema `all_gather_into_tensor` Evaluator

> **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`.

Reproduces the sealed metrics for `docs/components/pytorch-dist-all-gather-object-pickle-cuda-sync-stall-selection-faq.md`:
- **Tier 1 (`dist.all_gather_object` on an `NCCL` ProcessGroup)**: `14.80 ms/step` stall across 64 ranks due to `pickle.dumps`, 2 dynamic GPU `uint8` tensor allocations, 2 NCCL `AllGather` collectives (sizes + padded payload), and 2 hard `.cpu()` GPU-to-CPU synchronization barriers.
- **Tier 2 (`dist.all_gather_object` with an unstripped CUDA tensor inside the Python dict)**: `185.00 ms/step` stall and `cuda:0` memory spike as `pickle.loads` deserializes CUDA storages onto the default device.
- **Tier 3 (`dist.all_gather_object` over a dedicated CPU `gloo` group)**: `0` GPU stream syncs, `3.40 ms/step` latency (still bounded by 2 TCP socket rounds and `pickle`).
- **Tier 4 (Fixed-Schema Pre-Allocated Tensor Packing via `dist.all_gather_into_tensor` / SHM Ring)**: `0.08 ms/step` (`185.0x` faster than Tier 1, `42.5x` faster than Tier 3), `1` single collective, `0` `pickle` overhead, and `0` GPU stream sync barriers.
