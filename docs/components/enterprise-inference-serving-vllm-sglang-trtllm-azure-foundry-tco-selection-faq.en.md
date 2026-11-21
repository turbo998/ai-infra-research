# Enterprise Multi-Tenant LLM Inference Serving Stack Selection & Fully-Loaded TCO Guide: `vLLM V1` vs. `SGLang (RadixAttention)` vs. `TensorRT-LLM` vs. `Azure AI Foundry Managed Endpoints`

> **Language / 语言**: **English (Current)** | [中文](./enterprise-inference-serving-vllm-sglang-trtllm-azure-foundry-tco-selection-faq.md)  
> **Scope**: A practical decision and TCO guide for Cloud & AI Solution Architects—evaluating **`vLLM V1`, `SGLang`, `TensorRT-LLM`, and `Azure AI Foundry Managed / Serverless Endpoints` across (1) fully-loaded TCO including SRE engineering headcount (`$180K/year/FTE`), (2) multi-turn Agent prefix reuse (`RadixAttention`) and `DeepSeek-671B MoE` throughput, (3) Day-0 model onboarding velocity and multi-vendor silicon portability (`NVIDIA / AMD MI325X / Cloud TPU`), and (4) the exact GPU fleet size where self-hosting beats managed cloud endpoints**.  
> **Provenance**: Analytical performance and fully-loaded TCO model synthesized from published serving benchmarks and cloud pricing (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). See [`experiments/enterprise_inference_serving_vllm_sglang_trtllm_azure_foundry_tco_selection/results.json`](../../experiments/enterprise_inference_serving_vllm_sglang_trtllm_azure_foundry_tco_selection/results.json).

![Enterprise Inference Serving Stack Selection & Fully-Loaded TCO Summary](../../assets/svg/enterprise-inference-serving-vllm-sglang-trtllm-azure-foundry-tco-selection-faq-summary.svg)

> 🔗 **Interactive H5 Learning Lab & Companion Tools (Zero External Dependencies)**:
> - 📱 **[Interactive H5 Study Page: vLLM V1 vs. SGLang vs. TRT-LLM vs. Azure Foundry Fully-Loaded TCO Lab (`docs/h5/vllm-sglang-trtllm-azure-foundry-serving-lab.html`)](../h5/vllm-sglang-trtllm-azure-foundry-serving-lab.html)**
> - 🎓 **[AI Infra Panorama H5 Learning Hub (`docs/h5/index.html`)](../h5/index.html)**
> - 🧮 **[Interactive Serving Engine & SRE TCO Sizer (`enterprise-inference-serving-vllm-sglang-trtllm-azure-foundry-tco-selection-sizer.html`)](../reports/enterprise-inference-serving-vllm-sglang-trtllm-azure-foundry-tco-selection-sizer.html)**
> - 🃏 **[7-Card 4:5 Visual Report (`enterprise-inference-serving-vllm-sglang-trtllm-azure-foundry-tco-selection-cards.html`)](../reports/enterprise-inference-serving-vllm-sglang-trtllm-azure-foundry-tco-selection-cards.html)**

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Enterprise serving engine selection must account for both GPU efficiency and **SRE engineering headcount (`~$180,000/year per FTE`, equivalent to the annual lease of `6–8 H200 GPUs`)**—(1) For fleets of **`<= 8–12 GPUs`**, **`Azure AI Foundry Managed / Serverless Endpoints` (Tier 1)** wins on total TCO by eliminating `$180K–$360K` in dedicated K8s/GPU engineering salaries and onboarding models in `15 minutes`; (2) Above **`>= 16 GPUs`**, self-hosted open-source engines cross the TCO breakeven point, with **`vLLM V1` (Tier 3)** serving as the universal standard for `100+` models and multi-vendor silicon (`NVIDIA / AMD / TPU`) and **`SGLang` (Tier 4)** leading on multi-turn Agents (`RadixAttention` tree cache) and `DeepSeek-671B / Qwen3 MoE` (`FlashMLA + DeepEP + TBO`). Deploying **Tier 4 (Dual-Engine K8s: `SGLang` for MoE/Agents + `vLLM V1` for general/multimodal models + `Azure Foundry` spillover)** on a 32-GPU fleet achieves **`2,480 tok/s/GPU`** and cuts fully-loaded cost per 1M tokens by **`62.9%` (`$0.62 -> $0.23`)**!**

| Serving Stack Option (32-GPU Enterprise Fleet) | Core Architecture & Ecosystem Fit | New Model Onboarding | Agentic Goodput / GPU | Required SRE `FTE` | Fully-Loaded `$/1M` | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🟨 Tier 1: `Azure AI Foundry Managed`** | Turnkey managed MaaS/dedicated endpoints + built-in SLA | **`15 mins` 🏆** | `1,450 tok/s/GPU` | **`0.2 FTE` 🏆** | `$0.62 / 1M` | **Cheapest total TCO for `<=8–12 GPUs`** |
| **🟥 Tier 2: Pure `TensorRT-LLM`** | Ahead-of-time compiled C++ engines tied to GPU SKU | `5–10 days` ✕ | `2,320 tok/s/GPU` | **`2.5 FTE` (`$450K/yr`) ✕** | `$0.41 / 1M` | Only for `>128-GPU` single-model teams |
| **🟦 Tier 3: Standardized `vLLM V1`** | ZMQ multiprocess + Piecewise CUDA Graph + 100+ models | **`Day-0 (1 hr)` ✅** | `2,140 tok/s/GPU` ✅ | `0.8 FTE` ✅ | `$0.29 / 1M` ✅ | **Universal multi-model & multi-silicon standard** |
| **🟩 Tier 4: `SGLang + vLLM V1` Dual-Engine + `Foundry` Spillover (Recommended)** | **`SGLang` for MoE/Agents + `vLLM V1` for General + Cloud Backup** | **`Day-0 (1 hr)` 🏆** | **`2,480 tok/s/GPU` 🏆** | **`1.0 FTE` 🏆** | **`$0.23 / 1M` 🏆 (`-62.9%`)** | **Production standard for `>=16-GPU` platforms** |

---

## 2. 🎯 Architect Deep-Dive Q&A & Fully-Loaded TCO Math

### Q1: Why does self-hosting `TensorRT-LLM` on an 8-GPU cluster cost more per token than `Azure AI Foundry` once SRE salaries are included?
$$\text{Fully-Loaded Annual TCO} = N_{\text{GPU}} \times \text{Cost}_{\text{GPU/yr}} + N_{\text{SRE\_FTE}} \times \$180,000, \qquad \text{Effective \$/1M} = \frac{\text{TCO}_{\text{annual}}}{\text{Goodput}_{\text{cluster}} \times 31.536\text{M} \times U_{\text{avg}}}$$
- On an `8-GPU` cluster (`~$220K/yr` hardware), hiring `2` full-time engineers (`$360K/yr`) to compile and maintain custom `TensorRT-LLM` engines makes engineering labor `62%` of total cost (`$580K/yr`). Using `Azure AI Foundry Managed Endpoints` (or `AKS KAITO`) with `0.2 FTE` costs `~$326K/yr` all-in—saving **`$254K/year`**! Self-hosting only beats managed cloud endpoints once fleet size reaches **`>= 16 GPUs`** using low-ops `vLLM V1 / SGLang` containers (`<= 1.0 FTE`).

### Q2: How should architects divide workloads between `vLLM V1` and `SGLang` inside a Kubernetes cluster?
- **Use `SGLang` for Multi-Turn Agents, Structured JSON (`XGrammar`), and Large MoEs (`DeepSeek-671B / Qwen3`)**: Native `RadixAttention` prefix trees + `DeepEP` dual-batch overlap deliver `15%–30%` higher agentic Goodput.
- **Use `vLLM V1` for Broad Model Coverage (`100+` LLMs/VLMs/Embedding/Rerankers) and Multi-Vendor Hardware (`NVIDIA / AMD MI325X / Cloud TPU`)**: Day-0 HuggingFace compatibility with zero-bubble ZMQ multiprocessing and Piecewise CUDA Graphs.

---

## 3. 🧭 Serving Stack Selection Decision Tree

```mermaid
flowchart TD
    A["Select Enterprise Inference Serving Stack"] --> B{"Fleet Size & Dedicated AI Infra SREs?"}
    B -->|"<= 8–12 GPUs OR 0 Dedicated SREs"| C["🟨 Tier 1: Azure AI Foundry Managed / Serverless\nZero Ops Overhead · 15-Min Deployment"]
    B -->|">= 16 GPUs with K8s Platform Team"| D{"Workload Profile?"}
    D -->|"Multi-Turn Agents / DeepSeek-671B MoE"| E["🟩 SGLang Pool (RadixAttention + DeepEP + MTP)"]
    D -->|"100+ Models / Multimodal VLM / AMD & TPU"| F["🟦 vLLM V1 Pool (Piecewise Graph + Multi-Silicon)"]
```

---

## 4. 📊 4-Engine Enterprise Comparison Matrix

| Dimension | `vLLM V1` | `SGLang` | `TensorRT-LLM` | `Azure AI Foundry Managed` |
| :--- | :--- | :--- | :--- | :--- |
| **Day-0 Model Coverage** | **100+ Models (Day-0)** 🏆 | Mainstream LLMs/MoEs/VLMs | Slow (Requires Engine build) | Turnkey Model Catalog |
| **Multi-Turn Prefix Reuse** | Block-Hash APC | **`RadixAttention` Prefix Tree** 🏆 | KV Block Reuse | Managed Prompt Cache |
| **MoE & DeepSeek-671B** | Strong (`FlashMLA + DeepEP`) | **Industry Reference (`TBO + DeepEP`)** 🏆 | Strong (`Blackwell NVFP4`) | Managed Endpoint |
| **Hardware Portability** | **NVIDIA / AMD / TPU / Gaudi** 🏆 | NVIDIA / AMD ROCm | **NVIDIA Only** | Cloud-abstracted |
| **Engineering Ops Load** | Low (`~0.8 FTE`) | Low (`~0.8 FTE`) | **High (`2.5 FTE`)** ✕ | **Near-Zero (`0.2 FTE`)** 🏆 |

---

## 5. 🛠️ Production Readiness Checklist

- [ ] **Include SRE Salaries in Every TCO Model**: Stay on `Azure AI Foundry Managed / Serverless` or `AKS KAITO` below `12 GPUs` unless dedicated platform SREs already exist.
- [ ] **Standardize on OpenAI-Compatible `SGLang + vLLM V1` K8s Deployments**: Route MoE/Agent traffic to `SGLang` and multimodal/general models to `vLLM V1` behind `Envoy AI Gateway (K8s GIE)`.
