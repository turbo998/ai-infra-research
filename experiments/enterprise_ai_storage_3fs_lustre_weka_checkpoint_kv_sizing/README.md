# Experiment #596: Enterprise AI Storage (DeepSeek 3FS vs. Lustre/WEKA) & 2.8s Async Checkpoint Sizer

Reproduces the `85.7x` checkpoint stall reduction (`2.8s` Two-Phase Async DCP to Pinned Host DRAM vs. `240.0s` synchronous `10 GB/s` NFS write for a `2.4 TB` checkpoint across `256` GPUs) and `1,760 GB/s` aggregate throughput from pooling `32` compute nodes' local Gen5 NVMe drives via `DeepSeek 3FS` (`docs/components/enterprise-ai-storage-3fs-lustre-weka-checkpoint-kv-sizing-faq.md`).
