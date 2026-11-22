# Enterprise Fractional GPU Partitioning (`Hardware MIG 1g.18gb` vs. `CUDA MPS` vs. `Time-Slicing`) for Small AI Microservices (`#481`)

Reproducible analytical model comparing Whole-GPU Exclusive Allocation (`14 GPUs`), K8s GPU Time-Slicing (`+310% P99 jitter`), CUDA MPS (shared fault domain), and Hardware `MIG` (`7x 1g.18gb` slices per `H200`, `1.2%` jitter, full hardware fault/OOM isolation, `-85.7%` GPU cost).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_gpu_partitioning_mig_mps_timeslicing_small_model_packing_selection/run.py --check`
