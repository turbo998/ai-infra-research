# How Many GPUs to Fit & Run an LLM? The 3-Watermark Sizing Methodology (HBM Capacity × Bandwidth TPOT × Compute/Concurrency) & `H100 / H200 / B200` Guide

> **Language**: [中文](./llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects, Capacity Planners, and LLM Platform Engineers  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![3-Watermark LLM GPU Capacity Sizing Summary](../../assets/svg/llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Never size production GPUs by multiplying model parameters by 2 bytes! Enterprise GPU capacity sizing must satisfy all three watermarks simultaneously: $$N_{\text{GPU}} = \max\Big(\underbrace{N_{\text{mem}}}_{\text{Watermark 1: Fits in HBM}},\ \underbrace{N_{\text{bw}}}_{\text{Watermark 2: Meets TPOT SLA}},\ \underbrace{N_{\text{compute}}}_{\text{Watermark 3: Meets TTFT & QPS}}\Big)$$ Remember two golden architectural laws: (1) For MoE models (`DeepSeek-V3/R1 671B`), size Watermark 1 (HBM capacity) using TOTAL parameters (`671B`), but size Watermark 2 (Decode bandwidth TPOT) and Watermark 3 (Prefill FLOPs) using ACTIVE parameters (`37B`); (2) "Fits in VRAM $\neq$ Runs Fast"—a single `H200 (141 GB)` fits `Llama-3.3-70B FP8 (70 GB)`, but at `B=32, S=8K` its TPOT degrades to `32.4ms`, requiring `TP=2` or `TP=4` to aggregate HBM bandwidth and meet a `<20ms` interactive SLA!**

- **Interactive 3-Watermark GPU Sizer**: [Open Interactive Sizer (`llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-sizer.html`)](../reports/llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-cards.html`)](../reports/llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-cards.html)

---

## 2. 🎯 Architect Deep-Dive Q&A

### Q1: How do we calculate Watermark 1 (Fits in HBM: $N_{\text{mem}}$), and how much KV Cache do `MHA`, `GQA`, and `MLA` consume?
Divide total replica memory by **`0.85`** (reserving 15% safety headroom for allocator fragmentation and runtime buffers):
$$N_{\text{mem}} = \left\lceil \frac{M_{\text{weights}} + M_{\text{KV}}(B, S) + M_{\text{act}} + M_{\text{cudagraph}}}{0.85 \times \text{HBM}_{\text{per\_GPU}}} \right\rceil_{\text{next valid } TP \times PP}$$
1. **Model Weights ($M_{\text{weights}} = P_{\text{total}} \times \text{bytes\_per\_param}$)**: `2.0 B` (`BF16`), **`1.0 B` (`FP8`)**, or `0.55 B` (`INT4 AWQ`).
2. **KV Cache ($M_{\text{KV}}$)**:
   - **Standard `MHA / GQA`**: $M_{\text{KV}} = B \times S \times (2 \times L \times n_{kv} \times d_h \times \text{bytes}_{\text{KV}})$.
     - `Llama-3.3-70B` ($L=80, n_{kv}=8, d_h=128$): `327.68 KB/tok` (`BF16`) or **`163.84 KB/tok` (`FP8`)**.
     - `Llama-3.1-405B` ($L=126, n_{kv}=8, d_h=128$): `516.1 KB/tok` (`BF16`) or **`258.05 KB/tok` (`FP8`)**.
   - **`MLA` (`DeepSeek-V3/R1 671B`, $L=61, d_c+d_r=576$)**: $M_{\text{KV}} = B \times S \times (L \times 576 \times \text{bytes}_{\text{KV}})$ $\implies$ **`35.14 KB/tok` (`FP8`)** (`4.7x` smaller than GQA-70B!).

### Q2: How do we calculate Watermark 2 (Runs Fast Enough: Bandwidth TPOT Floor $N_{\text{bw}}$)?
$$\text{TPOT}_{\min} = \frac{M_{\text{active\_weights}} + M_{\text{KV\_active}}}{N_{\text{GPU}} \times BW_{\text{HBM}} \times \text{MBU}} \quad (\text{MBU} \approx 0.72)$$
Even if 1 GPU fits the weights, if `TPOT_min > TPOT_SLA`, you must scale `N_GPU` (`TP=2, 4, 8`) to aggregate HBM memory bandwidth.

---

## 3. 📊 Hardware Reference & 3-Model Sizing Matrix (`16K` Context, `TPOT <= 20ms`)

| Model & Precision | Weights ($M_{\text{weights}}$) | KV per `16K` Session | `H100 (80GB, 3.35 TB/s)` | `H200 (141GB, 4.8 TB/s)` | `B200 (192GB, 8.0 TB/s)` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Llama-3.3-70B` (`FP8`, GQA)** | `70.0 GB` | `2.62 GB` | `2x H100 (TP=2)` (`~20` sessions) | **`2x H200 (TP=2)` (`60+` sessions, `16.2ms` TPOT)** | `1x–2x B200` (`14.0ms` TPOT on 1 GPU!) |
| **`Llama-3.1-405B` (`FP8`, GQA)** | `405.0 GB` | `4.13 GB` | `8x H100 (TP=8)` (`~27` sessions) | **`8x H200 (TP=8)` (`125` sessions, `14.5ms` TPOT)** | **`4x B200 (TP=4)` (`55` sessions, `13.2ms` TPOT)** |
| **`DeepSeek-R1 671B` (`FP8`, MLA)** | `671.0 GB` (`37B` Active) | **`0.56 GB` (MLA!)** | `16x H100 (2 Nodes)` (8x80GB=640GB cannot fit!) | **`8x H200 (1 Node)` (`192` sessions, `13.6ms` TPOT)** | **`8x B200 (1 Node)` (`450+` sessions, `7.8ms` TPOT)** |

---

## 4. 🛠️ Production Readiness Checklist

- [ ] **Enforce 3-Watermark Sizing**: Always compute `max(N_mem / 0.85, N_bw, N_compute)` before procuring GPUs.
- [ ] **Separate MoE Total vs. Active Parameters**: Use `P_total` for Watermark 1 (HBM capacity) and `P_active` for Watermark 2/3 (TPOT bandwidth & Prefill compute).
- [ ] **Prefer Single-Node `8x H200 / B200` over Multi-Node `16x H100` for `671B MoE`**: Avoids cross-node RDMA latency and cuts cluster TCO by `69.5%`.
