# Enterprise Kubernetes GPU Sharing (Full GPU vs. Time-Slicing vs. MIG vs. MIG+MPS) Evaluator (Round 389)

Deterministic analytical evaluator comparing four Kubernetes GPU sharing architectures for packing `28` auxiliary AI microservices (`Embeddings, Rerankers, Guardrails, ASR, OCR, 1B-7B SLMs`) onto `80GB` H100/A100 nodes:

1. **Tier 1 (`tier1_one_pod_per_full_gpu`)**: 1-Pod-Per-Full-GPU (`28` GPUs required, `11.0%` avg SM utilization, `25.0 ms` P99 latency, `$78,400/mo`).
2. **Tier 2 (`tier2_k8s_software_time_slicing`)**: Kubernetes Software Time-Slicing (`4` GPUs required, `54.0%` avg SM utilization, `14.5%` noisy-neighbor OOM blast-radius incident rate, `98.0 ms` P99 latency due to context-switch stalls, `$11,200/mo`).
3. **Tier 3 (`tier3_hardware_nvidia_mig_static`)**: Hardware Multi-Instance GPU (`NVIDIA MIG 1g.10gb / 2g.20gb`, `4` GPUs required, `62.0%` avg SM utilization, `0.0%` noisy-neighbor OOM rate, `26.5 ms` P99 latency, `$11,200/mo`).
4. **Tier 4 (`tier4_hybrid_mig_fault_domains_plus_cuda_mps`)**: Hybrid MIG Hardware Fault-Domains + Intra-Slice CUDA MPS Spatial Sharing + K8s DRA Packing (`3` GPUs required [`9.3x` fewer GPUs than Tier 1], `78.5%` avg SM utilization, `0.0%` noisy-neighbor OOM rate, `21.0 ms` P99 latency [`4.67x` faster than Time-Slicing], `$8,400/mo` [`-89.3%` cost vs. Tier 1]).

## Usage

```bash
python experiments/enterprise_gpu_sharing_mig_mps_timeslicing_selection/run.py
python experiments/enterprise_gpu_sharing_mig_mps_timeslicing_selection/run.py --check
```
