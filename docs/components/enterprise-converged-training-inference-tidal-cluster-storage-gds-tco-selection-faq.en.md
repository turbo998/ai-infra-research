# Milestone #460: Enterprise Converged Training-Inference Tidal Cluster (`Daytime PD Serving + Nighttime GRPO/SFT Backfill`), Two-Stage Async Checkpointing & High-Speed Storage (`3FS / Lustre / WEKA + GDS`) Guide

> **Language / 语言**: [简体中文](./enterprise-converged-training-inference-tidal-cluster-storage-gds-tco-selection-faq.md) | **English**  
> **Companion Interactive Study & Sizing Tools**: [📱 Interactive H5 Study Lab](../h5/milestone-460-converged-tidal-cluster-storage-gds-tco-lab.html) · [🧮 HTML Sizer](../reports/enterprise-converged-training-inference-tidal-cluster-storage-gds-tco-selection-sizer.html) · [🃏 7-Card Visual Deck](../reports/enterprise-converged-training-inference-tidal-cluster-storage-gds-tco-selection-cards.html) · [📊 Reproducible Model](../../experiments/enterprise_converged_training_inference_tidal_cluster_storage_gds_tco_selection/) · [🎓 Back to AI Infra H5 Hub](../h5/index.html)

![Milestone #460 Converged Training-Inference Tidal Cluster & Storage Summary](../../assets/svg/enterprise-converged-training-inference-tidal-cluster-storage-gds-tco-selection-faq-summary.svg)

---

## 1. Executive Summary (3-Minute Takeaway)

> **Bottom-Line Recommendation**: Splitting 128 GPUs into physically siloed inference and training clusters with synchronous network checkpointing (`Tier 1`) strands **`61.5%` of cluster capacity** overnight and stalls training for **`225.0 seconds`** per `9 TB` checkpoint (`$4.56M/yr` TCO).
> - **Recommended Production Architecture (`Tier 4: Kueue TAS Converged Tidal Cluster + 2.8s Level-1 Sleep Serving Switch + 2.1s Two-Stage Async Host-DRAM Checkpoint + Hyperconverged DeepSeek 3FS + GPUDirect Storage`)**:
>   1. **4.9-Second Total Preemption Handover**: When morning online traffic surges, nighttime `GRPO / SFT` jobs flush `9 TB` of state to local `Host Pinned DRAM` in **`2.1 s`** and yield GPUs, while sleeping inference replicas wake up in **`2.8 s`**;
>   2. **Hyperconverged `3FS + GDS`** pools compute nodes' internal NVMe drives over RDMA at **`2.34 TB/s`**, lifting 24-hour fleet utilization from `38.5%` to **`93.4%`** and cutting annual TCO by **`63.2%` (`$1.68M/yr` vs. `$4.56M/yr`, saving `$2.88M/yr`)**.

| Dimension | `Tier 1: Siloed Clusters + Sync NFS/Ceph` | `Tier 2: Shared K8s (Pod Restart) + Lustre` | `Tier 3: Kueue Preemption + Async Ckpt + WEKA` | `Tier 4: Converged Tidal (2.8s Sleep) + 3FS GDS + 2.1s Ckpt (🏆)` |
|---|---|---|---|---|
| **24-Hour Effective Fleet Utilization** | **`38.5%` ✕** | `64.0%` | `82.5%` ✓ | **`93.4%` (Daytime PD + Night Backfill)** 🏆 |
| **9 TB Checkpoint GPU Stall Time** | **`225.0 s` ✕** | `42.0 s` | `3.1 s` ✓ | **`2.1 s` (D2H Pinned DRAM stage)** 🏆 |
| **Tidal Role Handover Latency** | `540.0 s` ✕ | `95.0 s` | `18.5 s` ✓ | **`4.9 s` (`2.1s` Ckpt + `2.8s` Wakeup)** 🏆 |
| **Annual 128-GPU Cluster TCO** | `$4,560,000 / yr` ✕ | `$3,180,000 / yr` | `$2,340,000 / yr` ✓ | **`$1,680,000 / yr` (Saves $2.88M/yr)** 🏆 |

---

## 2. Provenance & Reproducibility

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)` — verified via [`experiments/enterprise_converged_training_inference_tidal_cluster_storage_gds_tco_selection/run.py`](../../experiments/enterprise_converged_training_inference_tidal_cluster_storage_gds_tco_selection/run.py).
