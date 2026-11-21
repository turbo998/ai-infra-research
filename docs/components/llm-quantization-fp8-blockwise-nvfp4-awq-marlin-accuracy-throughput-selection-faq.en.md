# LLM Low-Bit Quantization Architecture Selection: `BF16` vs. Fine-Grained `Block FP8 (1x128 & 128x128)` vs. `W4A16 AWQ Marlin` vs. `Blackwell NVFP4` Accuracy, Throughput & Hardware Guide

> **Language**: [中文](./llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and LLM Serving Engineers  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![LLM Quantization Selection Summary](../../assets/svg/llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Match quantization to (1) GPU hardware generation, (2) active batch size $B$, and (3) scaling granularity: on `A100` (which lacks native FP8 TensorCores) or at low concurrency ($B \le 16$), choose `W4A16 AWQ + Marlin` (`0.55 Bytes/param`, `11.8ms` low-batch TPOT); on `Hopper (H100 / H200)` and `Ada (L40S)` at medium/high concurrency ($B \ge 24$), choose fine-grained `Block FP8 W8A8KV8` (`1x128` per-token activation tiles × `128x128` weight blocks) to isolate SwiGLU/MoE outliers—retaining `99.8%` benchmark accuracy while delivering `2,140 Goodput tok/s/GPU` (`2.18x` vs. BF16, `-60.9%` `$/1M`); and on `Blackwell (B200 / GB200)`, adopt native `NVFP4` (`16-element` micro-block FP8 scales, `0.5625 Bytes/param`)!**

- **Interactive Quantization & Hardware Matcher**: [Open Interactive Sizer (`llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-sizer.html`)](../reports/llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-cards.html`)](../reports/llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-cards.html)

---

## 2. 📊 4-Tier Quantization Comparison Matrix (`H200 / B200`)

| Metric | Tier 1: Unquantized `BF16 (W16A16KV16)` | Tier 2: Naive Static `Per-Tensor FP8` | Tier 3: `W4A16 AWQ + Marlin` | Tier 4 (Recommended): Fine-Grained `Block FP8 (1x128 × 128x128)` + `FP8 KV` |
| :--- | :--- | :--- | :--- | :--- |
| **Bytes per Weight Parameter** | `2.00 Bytes` | `1.00 Byte` | **`0.55 Bytes`** | **`1.00024 Bytes`** |
| **Relative Accuracy Retention** | `100.0%` | `94.2%` ✕ (`-5.8%`) | `98.4%` ✓ | **`99.8%` ✓ (Near-Lossless)** |
| **SLA-Compliant Goodput / GPU** | `980 tok/s` | `1,950 tok/s` | `1,540 tok/s` | **`2,140 tok/s` ✓ (`2.18x`)** |
| **Cost per 1M Tokens** | `$0.92 / 1M` | `$0.46 / 1M` | `$0.54 / 1M` | **`$0.36 / 1M` ✓ (`-60.9%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Never Run `FP8` on `A100`**: Use `W4A16 AWQ Marlin` or `W8A8 INT8 SmoothQuant` on Ampere GPUs.
- [ ] **Require Fine-Grained `1x128 / 128x128` Block Scaling for `FP8`** on `H100 / H200` to preserve math/coding reasoning accuracy.
