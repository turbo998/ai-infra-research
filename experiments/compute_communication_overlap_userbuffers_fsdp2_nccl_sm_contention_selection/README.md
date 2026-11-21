# Compute-Communication Overlap & NCCL SM-Contention Governance Experiment (`TransformerEngine Userbuffers` vs. `FSDP2 Async` vs. `Standard NCCL`)

Reproducible analytical model comparing 4 compute-communication overlap architectures on `H100 / H200 (132 SMs)` running `TP=8 + FSDP2`:
- **Tier 1 (`Sequential Non-Overlapped GEMM -> NCCL`)**: `32.0%` exposed communication stall, `38.5%` MFU.
- **Tier 2 (`Naive Multi-Stream GEMM + Standard NCCL`, `28 SMs` stolen)**: `21.5%` GEMM wave-quantization slowdown, `14.5%` net overhead, `46.2%` MFU.
- **Tier 3 (`FSDP2 Async Prefetch` + `NVLS Hardware Offload`, `6 SMs`)**: `4.2%` exposed overhead, `53.8%` MFU.
- **Tier 4 (`TransformerEngine Userbuffers + TMA Copy Engine`, `2 SMs`, Recommended)**: `0.8%` exposed communication overhead, `57.4%` MFU (`+49.1%` throughput over Tier 1).

## Run & Verify

```bash
python experiments/compute_communication_overlap_userbuffers_fsdp2_nccl_sm_contention_selection/run.py
python experiments/compute_communication_overlap_userbuffers_fsdp2_nccl_sm_contention_selection/run.py --check
```
