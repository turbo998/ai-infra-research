# LLM Training Memory Anatomy (`16Ψ` State + Activations) & Sharding Selection: `ZeRO-1 / ZeRO-2 / ZeRO-3 (PyTorch FSDP2)` vs. `Megatron 3D Parallelism` & Selective Activation Checkpointing Guide

> **Language**: [中文](./llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Distributed Training Platform Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![LLM Training 16Ψ Memory & FSDP2 vs Megatron 3D Summary](../../assets/svg/llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Mixed-precision (`BF16 + FP32 AdamW`) full-parameter training requires a baseline of `16Ψ` bytes of static state before a single token of activations (`2Ψ` BF16 weights + `2Ψ` BF16 gradients + `4Ψ` FP32 master weights + `4Ψ` FP32 momentum $m$ + `4Ψ` FP32 variance $v = 16\text{ Bytes/param}$, or `1,120 GB` for a `70B` model). Never run flat global `ZeRO-3` across large multi-node clusters (`42.5%` cross-node communication stall drops MFU to `31.8%`). Choose `PyTorch FSDP2 Hybrid Sharding (HSDP: 8-way NVLink ZeRO-3 × inter-node ZeRO-1)` for `<=70B` SFT/post-training, and `Megatron 3D Parallelism (Intra-Node TP × Interleaved 1F1B PP × ZeRO-1 DP)` paired with `Selective Activation Checkpointing (SAC)` for `>=70B–671B` pre-training—achieving `58.8%` MFU (`+27.0 pts` vs. Tier 1) and cutting cost per 1T training tokens by `46.0%` (`$262K`)!**

- **Interactive `16Ψ` Training Memory & 3D Sizer**: [Open Interactive Sizer (`llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism-sizer.html`)](../reports/llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism-selection-cards.html`)](../reports/llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism-selection-cards.html)

---

## 2. 📊 4-Tier Distributed Training Comparison Matrix (`128x H200`, `Llama-3.3-70B`)

| Metric | Tier 1: Flat Global `ZeRO-3` (128 GPUs) + Full Activation Ckpt | Tier 2: `PyTorch FSDP2 HSDP` (`8×16`) | Tier 3: `Megatron 3D (TP8×PP4×DP4)` + Full Activation Ckpt | Tier 4 (Recommended): `TP8 × 1F1B PP × ZeRO-1 DP` + `SAC` + `FP8` |
| :--- | :--- | :--- | :--- | :--- |
| **Achieved Training `MFU`** | `31.8%` ✕ | `49.4%` ✓ | `46.2%` | **`58.8%` ✓ (`+27.0 pts`)** |
| **Cross-Node Comm Share** | `42.5%` ✕ | `14.2%` | `9.8%` | **`6.4%` ✓** |
| **Per-GPU Peak HBM (`141GB` max)** | `54.0 GB` | `128.5 GB` | `72.0 GB` | **`94.0 GB` ✓** |
| **Cost per 1T Training Tokens** | `$485,000` | `$312,000` | `$334,000` | **`$262,000` ✓ (`-46.0%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Size Static State via `16Ψ` Equations**: Verify per-GPU static memory is `<=40%–50%` of HBM capacity.
- [ ] **Confine Weight Sharding (`TP / HSDP`) to Intra-Node NVLink** and use `ZeRO-1` across nodes.
- [ ] **Use `Selective Activation Checkpointing (SAC)` + `Sequence Parallelism (SP)`** instead of Full Checkpointing.
