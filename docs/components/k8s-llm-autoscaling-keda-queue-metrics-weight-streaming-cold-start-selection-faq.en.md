# Kubernetes LLM Autoscaling & Cold-Start Optimization Guide: `KEDA` Queue/KV Triggers, `Run:ai Model Streamer / NVMe GDS` Weight Streaming & `vLLM Sleep-Mode` Instant Wake

> **Language**: [中文](./k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Kubernetes AI Platform Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![K8s LLM Autoscaling KEDA & Weight Streaming Cold-Start Summary](../../assets/svg/k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Two classic Kubernetes anti-patterns break LLM autoscaling: (1) scaling HPA on `DCGM_FI_DEV_GPU_UTIL`—because continuous batching keeps SMs active `85%–95%` of the time even at `Batch=1` (`2%` KV usage), causing `72%` false/late scaling; and (2) baking `70–671 GB` model weights into OCI container images or loading `.safetensors` sequentially over NFS (`640s` cold start!). Standardize on `KEDA` Prometheus triggers (`vllm:num_requests_waiting > 4` OR `vllm:gpu_cache_usage_perc > 0.78`), decouple `<8 GB` runtime images from weights, and load weights via `Run:ai Model Streamer` (`24s` cold pod from S3/GCS/Blob) or local NVMe `GPUDirect Storage (GDS)` + `vLLM Sleep-Mode level=1` (`4.2s` wake from Pinned Host DDR5)—cutting scale-out ready time `51.2x` (`640s -> 12.5s`) and reducing monthly elastic cluster TCO by `64.1%` (`$245K -> $88K/mo`)!**

- **Interactive K8s Cold-Start & KEDA Sizer**: [Open Interactive Sizer (`k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-sizer.html`)](../reports/k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-cards.html`)](../reports/k8s-llm-autoscaling-keda-queue-metrics-weight-streaming-cold-start-selection-cards.html)

---

## 2. 📊 4-Tier K8s LLM Autoscaling & Cold-Start Comparison Matrix (`70B LLM`)

| Metric | Tier 1: `GPU_UTIL` HPA + `140GB` Fat Image | Tier 2: `KEDA` Queue/KV + Shared NFS Read | Tier 3: `KEDA` + `Run:ai Model Streamer` (S3/Blob) | Tier 4 (Recommended): `KEDA` + Local `NVMe GDS` + `vLLM Sleep-Mode (level=1)` |
| :--- | :--- | :--- | :--- | :--- |
| **Scale-Out Replica Ready Time** | `640.0 s` (10.7m ✕) | `165.0 s` | `24.0 s` ✓ | **`12.5 s` ✓ (`4.2s` Sleep Wake!)** |
| **Burst Traffic P99 `TTFT`** | `18,500 ms` ✕ | `4,200 ms` ✕ | `920 ms` ✓ | **`540 ms` ✓** |
| **False / Late Scaling Rate** | `72.0%` ✕ | `8.5%` | `2.1%` | **`0.4%` ✓** |
| **Monthly Elastic Cluster TCO** | `$245,000 / mo` | `$158,000 / mo` | `$112,000 / mo` | **`$88,000 / mo` ✓ (`-64.1%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Ban `DCGM_FI_DEV_GPU_UTIL` for Inference HPA**: Scale via `KEDA` on `vllm:num_requests_waiting` and `vllm:gpu_cache_usage_perc`.
- [ ] **Decouple Container Images (`<8 GB`) from Model Weights** and use `--load-format runai_streamer` or local NVMe `GDS` caching.
- [ ] **Configure Graceful Connection Draining (`preStop` + `terminationGracePeriodSeconds: 180`)** so scale-in never drops active SSE streams.
