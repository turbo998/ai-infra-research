# Enterprise Kubernetes (AKS/EKS) LLM Autoscaling, KAITO/KEDA & Model Asset Fast Cold-Start (`enterprise_k8s_aks_kaito_keda_autoscaling_fast_load_selection`)

Deterministic architectural evaluator for **SA Guide**: comparing **Legacy HPA + 158GB Fat Image**, **KEDA + Shared NFS/BlobFuse**, **KAITO/KEDA + P2P Image + Local NVMe + Run:ai Streamer**, and **KEDA + Host-DRAM Sleep Mode + P2P RDMA Weight Broadcast**.

## Usage

```bash
python experiments/enterprise_k8s_aks_kaito_keda_autoscaling_fast_load_selection/run.py
python experiments/enterprise_k8s_aks_kaito_keda_autoscaling_fast_load_selection/run.py --check
```
