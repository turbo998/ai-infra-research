# Enterprise Kubernetes / AKS LLM Autoscaling & `11.5s` Cold-Start Sizer (`#526`)

Reproducible analytical model comparing **Legacy K8s HPA + Full Docker Pull + HF HTTP Download (`420s` cold start, `38.4%` burst drop rate, `32 GPUs` ✕)**, **Local NVMe HostPath `Safetensors mmap` (`48s` cold start)**, and **`AKS/K8s KAITO + KEDA Waiting-Queue Autoscaling + Dragonfly P2P OCI Streaming + GDS/RDMA Peer Weight Broadcast` (`11.5s` cold start [36.5x faster], `0.0%` drops, `16x H200 GPUs`, `-50.0%` GPUs 🏆)**.

## Run

```bash
python experiments/enterprise_k8s_aks_kaito_keda_dragonfly_gds_coldstart_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-k8s-aks-kaito-keda-dragonfly-gds-coldstart-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-k8s-aks-kaito-keda-dragonfly-gds-coldstart-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/k8s-aks-kaito-keda-dragonfly-gds-coldstart-lab.html](../../docs/h5/k8s-aks-kaito-keda-dragonfly-gds-coldstart-lab.html)
