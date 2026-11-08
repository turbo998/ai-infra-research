# AI Cluster Storage, Two-Phase Async Checkpointing & Tiered KV Cache Offloading Analytical Model

Deterministic analytical evaluation backing `docs/components/ai-storage-checkpoint-kv-offload-selection-faq.md` (`.en.md`) and `docs/reports/ai-storage-checkpoint-kv-offload-cards.html`.

## Provenance
- **Classification**: `synthetic_analytical_model_calibrated_to_anonymized_reported_specs` (`independently_reproduced=false`, `paid_cloud_gpu_launched=false`, `live_512gpu_cluster_measurement="not_measured"`).
- **Scenario A (512-GPU Training Checkpoint)**: `9,000 GB` (`9.0 TB`) distributed checkpoint every `1,800 s` (`30 min`). Compares synchronous POSIX (`40.0 GB/s` -> `225.0 s` stall, `87.50%` goodput), synchronous GDS over RDMA All-Flash (`360.0 GB/s` -> `25.0 s` stall, `98.61%` goodput), and Two-Phase Async Checkpointing (`2,880.0 GB/s` HBM->Pinned DRAM -> `3.125 s` stall, `72.0x` stall reduction, `99.83%` goodput).
- **Scenario B (64K-Token Multi-Turn Agent KV Cache Tiering)**: `2.048 GB` (`16.384 Gb`) FP8 KV state per session. Compares full Prefill recompute (`780.0 ms`, `$0.96/1M`), traditional POSIX/TCP disk (`652.0 ms`, `1.20x`), Shared RDMA/NVMe-oF pool with GDS (`62.0 ms`, `12.58x`, `$0.48/1M`, `-50.0%`), and Pinned Host DRAM over PCIe Gen5 (`52.0 ms`, `15.0x`, `$0.44/1M`, `-54.2%`).

## Usage
```bash
python experiments/ai_storage_checkpoint_kv_offload/run.py
python experiments/ai_storage_checkpoint_kv_offload/run.py --check
```
