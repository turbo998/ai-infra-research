# LLM Inference Serving Engine Selection Guide: `vLLM V1` vs. `SGLang (RadixAttention)` vs. `TensorRT-LLM` vs. `NVIDIA Dynamo` vs. `Ollama / llama.cpp`

> **Language**: [中文](./inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Enterprise AI Platform Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Inference Serving Engine Selection Summary](../../assets/svg/inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Match the serving engine to workload pattern, hardware fleet, and team ops maturity: (1) restrict `Ollama / llama.cpp` to laptops, edge devices, and single-user dev sandboxes (`B <= 4`); (2) choose `vLLM V1` as the default enterprise multi-model MaaS workhorse (broadest Day-0 model support across `NVIDIA + AMD MI300X + Google Cloud TPU`); (3) choose `SGLang` for large MoE models (`DeepSeek-V3/R1 EP=64 + DeepEP + Two-Batch Overlap`) and multi-turn Agent workflows (`RadixAttention` + `XGrammar`); (4) choose `TensorRT-LLM` when squeezing the last `10%–15%` low-latency speed out of a fixed flagship model on `Blackwell B200 (NVFP4)`; and (5) at cluster scale (`>=16 GPUs`), deploy `NVIDIA Dynamo` above `SGLang / vLLM V1 / TRT-LLM` workers to orchestrate KV-aware prefix routing and `NIXL` RDMA PD disaggregation—achieving `2,490 Goodput tok/s/GPU` and `92.8%` multi-turn KV hit rate (`$0.34/1M` tokens)!**

- **Interactive Inference Engine Matcher**: [Open Interactive Sizer (`inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-sizer.html`)](../reports/inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-cards.html`)](../reports/inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-cards.html)

---

## 2. 📊 5-Engine Architectural Comparison Matrix (`32x H200`)

| Dimension | `Ollama / llama.cpp` | `vLLM V1` | `SGLang` | `TensorRT-LLM` | `NVIDIA Dynamo` + (`SGLang / vLLM / TRT-LLM`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Role** | Edge / Laptop CLI (`GGUF`) | General Multi-Model MaaS Engine | Radix Agent & Large MoE Engine | AOT C++ Peak-Speed Engine | **Cluster KV Router & PD Control Plane** |
| **Goodput TPS / GPU** | `290 tok/s` ✕ (In DC) | `1,960 tok/s` ✓ | `2,280 tok/s` ✓ | `2,120 tok/s` ✓ | **`2,490 tok/s` ✓** |
| **Multi-Turn KV Hit Rate** | `15.0%` | `76.4%` | `85.2%` | `74.0%` | **`92.8%` ✓ (KV-Aware Router)** |
| **Hardware Portability** | CPU / Mac / Edge GPU | **NVIDIA + AMD + TPU** | NVIDIA + AMD | NVIDIA GPU Only | Engine-agnostic control plane |
| **Cost per 1M Tokens** | `$2.95 / 1M` ✕ | `$0.44 / 1M` | `$0.38 / 1M` | `$0.41 / 1M` | **`$0.34 / 1M` ✓ (`-88.5%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Never Use `Ollama` for Multi-Tenant Datacenter Production**: Standardize on `vLLM V1` (Dense/Multi-Model) and `SGLang` (MoE/Agents).
- [ ] **Place `NVIDIA Dynamo` (or `vLLM Production Stack` KV Router) in Front of Multi-Replica Pools** instead of stateless Round-Robin load balancers.
