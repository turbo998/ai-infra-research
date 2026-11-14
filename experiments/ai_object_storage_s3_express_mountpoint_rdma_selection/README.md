# AI Cloud-Native Object Storage High-Throughput Streaming (`boto3` vs. `Mountpoint FUSE` vs. `s5cmd / aws-crt` vs. `S3 Express + S3-over-RDMA`) Selection (`ai_object_storage_s3_express_mountpoint_rdma_selection`)

Deterministic Python 3.12 analytical evaluator for **Round 183**: comparing `Naive Python boto3 / Legacy s3fs-fuse` (Tier 1), `Modern Rust FUSE (Mountpoint for S3 + NVMe Cache)` (Tier 2), `Native C/Rust 128-Stream Parallel Range-GET (aws-c-s3 / s5cmd)` (Tier 3), and `S3 Express One Zone + S3-over-RDMA (cuObject)` (Tier 4).

## Reproduction

```bash
python -m experiments.ai_object_storage_s3_express_mountpoint_rdma_selection.run
python -m unittest tests/test_ai_object_storage_s3_express_mountpoint_rdma_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_cloud_storage_benchmarks`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`
