# 128K–1M Long-Context Tiered KV Cache Offload (`HBM + Host Pinned DRAM + 3FS/NVMe GDS`), 2D Context Parallelism (`Ulysses × Ring USP`) & `MLA / 7:1 Hybrid SSM` Sizing Guide

> **Language / 语言**: **English (Current)** | [中文](./long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-faq.md)  
> **Scope**: A quantitative architecture and sizing guide for Cloud & AI Solution Architects designing `128K–1M` long-context RAG and multi-turn coding/research agents—explaining **(1) why evicting a `128K` session from HBM triggers a `5,200 ms` $O(S^2)$ Prefill recompute stall whereas restoring it from Host Pinned DRAM over `8x PCIe Gen5` takes just `10–47 ms` (`110x–520x` faster), (2) how `DeepSeek MLA` (`35.14 KB/tok`) and `7:1 Hybrid SSM/GQA` (`4.4 KB/tok`) prevent `1M-token` HBM OOMs, and (3) how `2D Context Parallelism (Intra-Node Ulysses × Inter-Node Ring)` slashes uncached `256K` TTFT from `28.4 s` to `1.85 s`**.  
> **Provenance**: Analytical model synthesized from published long-context serving architectures (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). See [`experiments/long_context_tiered_kv_cache_hbm_dram_nvme_usp_mla_sizing_selection/results.json`](../../experiments/long_context_tiered_kv_cache_hbm_dram_nvme_usp_mla_sizing_selection/results.json).

![128K-1M Long-Context Tiered KV Offload, 2D USP & MLA/Hybrid Sizing Summary](../../assets/svg/long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-faq-summary.svg)

> 🔗 **Interactive H5 Learning Lab & Companion Tools (Zero External Dependencies)**:
> - 📱 **[Interactive H5 Study Page: 128K–1M Tiered KV Offload, Recompute vs. Restore Math & 2D USP Lab (`docs/h5/long-context-tiered-kv-usp-mla-lab.html`)](../h5/long-context-tiered-kv-usp-mla-lab.html)**
> - 🎓 **[AI Infra Panorama H5 Learning Hub (`docs/h5/index.html`)](../h5/index.html)**
> - 🧮 **[Interactive Long-Context KV Sizer (`long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-sizer.html`)](../reports/long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-sizer.html)**
> - 🃏 **[7-Card 4:5 Visual Report (`long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-cards.html`)](../reports/long-context-tiered-kv-cache-hbm-dram-nvme-usp-mla-sizing-selection-cards.html)**

---

## 1. 📌 Executive Summary

> **Bottom Line**: **On a 32-GPU cluster serving `128K-Context` multi-turn agents, relying solely on GPU HBM for KV caching (Tier 1) consumes **`20.97 GB`** per session under `70B GQA-8`, capping the entire fleet at `18` concurrent sessions and forcing a **`5,200 ms` full $O(S^2)$ Prefill recompute** whenever a user pauses for 15 seconds! Adopting **Tier 4 (`DeepSeek MLA / 7:1 Hybrid SSM` Attention + `Intra-Node Ulysses × Inter-Node Ring` 2D Context Parallelism + `L1 HBM -> L2 Pinned DRAM -> L3 RDMA/3FS NVMe GDS` 3-Tier KV Pool)** shrinks per-session `128K` KV footprint to **`4.50 GB (MLA)` or `0.58 GB (Hybrid)`**, slashes multi-turn restore `TTFT` by **`66.7x` (`5,200 ms -> 78 ms`)**, expands concurrent `128K` sessions by **`35.5x` (`18 -> 640`)**, and lowers cost per 1M tokens by **`91.6%` (`$2.85 -> $0.24`)**!**

| Architecture Tier (32-GPU Fleet · `128K` Agent) | Attention, Parallelism & KV Pool Stack | Multi-Turn Restore `TTFT` | Max Concurrent `128K` Sessions | Fleet Goodput | Cost / 1M Tokens | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🟥 Tier 1: HBM-Only + Round-Robin** | Evicted KV discarded; triggers full `128K` $O(S^2)$ recompute | **`5,200 ms` ✕** | **`18 Sessions` ✕** | `640 tok/s` ✕ | `$2.85 / 1M` ✕ | **Unusable at `128K+`**: Constant recompute stalls |
| **🟨 Tier 2: `GQA-8 FP8 KV` + Host `Pinned DRAM`** | Offloads warm KV over `8x PCIe Gen5` (`448 GB/s`) to `2TB` host RAM | `340 ms` | `96 Sessions` | `2,450 tok/s` | `$0.74 / 1M` | `15x` faster multi-turn TTFT on single nodes |
| **🟦 Tier 3: `2D USP (Ulysses×Ring)` + `Mooncake` RDMA** | 2D Context Parallelism + cluster-wide RDMA DRAM pool | `145 ms` ✅ | `280 Sessions` ✅ | `4,820 tok/s` ✅ | `$0.38 / 1M` ✅ | Standard for multi-node GQA-8 long context |
| **🟩 Tier 4: `MLA / Hybrid` + `2D USP` + 3-Tier KV Pool (Recommended)** | **`MLA (4.5GB) / Hybrid (0.58GB)` + `HBM/DRAM/3FS GDS` Pool** | **`78 ms` 🏆 (`66.7x`)** | **`640 Sessions` 🏆 (`35.5x`)** | **`6,940 tok/s` 🏆 (`10.8x`)** | **`$0.24 / 1M` 🏆 (`-91.6%`)** | **128K–1M Enterprise Long-Context Standard** |

---

## 2. 🎯 Architect Deep-Dive Q&A & First-Principles Long-Context Math

### Q1: Why is restoring KV Cache from Host DRAM/RDMA (`O(S)`) over 100x faster than recomputing Prefill (`O(S^2)`) at `128K–1M` context?
* **Per-Session FP8 KV Footprint at `128K` vs. `1M` Tokens**:
  - **Standard `GQA-8` (`Llama-3.3-70B`, `163.84 KB/tok`)**: **`20.97 GB` at `128K`** | **`167.77 GB` at `1M`** (exceeds a single `141 GB H200`!).
  - **`DeepSeek MLA` (`671B`, `35.14 KB/tok`)**: **`4.50 GB` at `128K`** (`4.67x` smaller) | **`35.98 GB` at `1M`**.
  - **`7:1 Hybrid SSM/GQA` (`Qwen3-Next / Jamba`, `4.40 KB/tok`)**: **`0.58 GB` at `128K`** (`36.2x` smaller) | **`4.52 GB` at `1M`**.
* **Recompute vs. PCIe Gen5 / RDMA Restore Equation**:
  $$T_{\text{recompute}}(S) \propto \frac{2\Psi_{\text{active}}S + 4 n_h d_h L S^2}{\text{TFLOPS}} \;\;(\mathbf{5,200\text{ ms}}\text{ at }128\text{K}), \qquad T_{\text{restore}}(S) = \frac{\text{KV}_{\text{bytes}}(S)}{\text{BW}_{\text{PCIe/RDMA}}} \;\;(\mathbf{10.0\text{–}46.8\text{ ms}}\text{ over }448\text{ GB/s})$$

### Q2: How does `2D Hybrid Context Parallelism (Ulysses × Ring USP)` accelerate uncached `128K–1M` prompts?
- **Intra-Node `DeepSpeed Ulysses SP` (over `900 GB/s NVLink`)**: Partitions the sequence and transposes via `All-to-All` into head-parallel execution (bounded by the number of KV heads, e.g., `Ulysses <= 8` for GQA-8).
- **Inter-Node `Ring Attention` (over `400G RDMA`)**: Streams $(K, V)$ chunks around a multi-node ring via asynchronous P2P RDMA completely overlapped with local blockwise FlashAttention compute—cutting uncached `256K` TTFT from `28.4 s` to **`1.85 s` (`15.3x` speedup)**.

---

## 3. 🧭 3-Tier Zero-Copy KV Cache Pyramid

```mermaid
flowchart LR
    HBM["🔥 L1: GPU HBM3e\nActive Decode (<1ms)"] <== "8x PCIe Gen5 (448 GB/s)\nRestore 128K MLA in 10ms" ==> DRAM["⚡ L2: Host Pinned DRAM & RDMA Pool\nWarm Multi-Turn Sessions (2–64 TB)"]
    DRAM <== "GPUDirect Storage (GDS)\nZero-Copy NVMe / 3FS" ==> NVME["💾 L3: NVMe / 3FS Parallel FS\nDurable Repo & Doc KV (PB Scale)"]
```

---

## 4. 📊 Long-Context Attention & KV Storage Tier Matrix

| Metric | `GQA-8` (`Llama-3.3-70B`) | `DeepSeek MLA` (`671B`) | `7:1 Hybrid SSM/GQA` (`Qwen3-Next`) |
| :--- | :--- | :--- | :--- |
| **FP8 KV Bytes / Token** | `163.84 KB/tok` | **`35.14 KB/tok` (`4.67x` smaller)** 🏆 | **`4.40 KB/tok` (`37.2x` smaller)** 🏆 |
| **`128K` Session KV Size** | `20.97 GB` | **`4.50 GB`** | **`0.58 GB`** |
| **`1M` Session KV Size** | `167.77 GB` ✕ | **`35.98 GB` ✓** | **`4.52 GB` ✓** |
| **PCIe Gen5 Restore (`128K`)** | `46.8 ms` (`111x` vs. recompute) | **`10.0 ms` (`520x` vs. recompute)** 🏆 | **`1.3 ms`** 🏆 |

---

## 5. 🛠️ Production Readiness Checklist

- [ ] **Enable L2 Host Pinned DRAM + L3 GDS Offload**: Configure `Mooncake` or `LMCache` to spill evicted HBM KV blocks into `Pinned Host DRAM` and `NVMe GDS` instead of discarding them.
- [ ] **Use `2D USP (Ulysses × Ring)` for Uncached `>=64K` Prompts**: Map `Ulysses SP` inside the NVLink domain (`<=8`) and `Ring Attention` across RDMA nodes.
- [ ] **Prefer `MLA` or `7:1 Hybrid SSM` Models for `256K–1M` Workloads**: Reduce KV memory pressure by `4.7x–37x` before buying additional GPUs.
