# Enterprise Confidential AI Computing (`NVIDIA H100/H200/Blackwell GPU TEE` + `PPCIE` Multi-GPU Encryption) Overhead & Compliance Sizer (`#469`)

Reproducible analytical model comparing Non-TEE PII Redaction, Synchronous CPU-GPU Bounce-Buffer TEE, Single-GPU `H200 141GB` TEE (`TP=1` Zero Inter-GPU Bounce), and Multi-GPU `PPCIE / Blackwell TEE-I/O` line-rate encrypted NVLink for regulated finance/healthcare workloads.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_confidential_ai_gpu_tee_h200_blackwell_ppcie_overhead_selection/run.py --check`
