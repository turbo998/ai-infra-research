# Enterprise AI Gateway Deep-Dive Selection Guide: `Azure API Management (AI Gateway)` vs. `LiteLLM Proxy` vs. `Kong AI Gateway` vs. `Envoy AI Gateway / Higress`

> **Language**: [中文](./enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Enterprise Platform Engineering Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Enterprise AI Gateway Selection Summary](../../assets/svg/enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Traditional HTTP Layer-7 proxies (`Nginx / ALB`) fail on LLM traffic because request-count (`RPM`) limits ignore `256x` context-length variance and SSE streaming hides output token counts until the final `[DONE]` chunk. Match your AI Gateway to your platform architecture: (1) choose `Azure API Management (AI Gateway)` for Azure-centric enterprises standardizing on `Azure OpenAI PTU + Entra ID RBAC`; (2) choose `LiteLLM Proxy` for rapid onboarding of `100+` model providers, virtual keys, and out-of-the-box team spend ledgers (`<=1,000 RPS`); (3) choose `Kong AI Gateway` when consolidating onto an existing Kong Enterprise footprint; and (4) choose `Envoy AI Gateway / Higress` (with the `K8s Gateway API Inference Extension`) for high-throughput (`>=1,500 RPS`) or self-hosted `vLLM / SGLang` clusters—delivering `0.9ms` P99 proxy overhead, `120ms` PTU-to-PAYGO failover, `0.0%` quota breaches, and `-64.2%` monthly platform TCO!**

- **Interactive AI Gateway Matcher & Quota Sizer**: [Open Interactive Sizer (`enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-sizer.html`)](../reports/enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-cards.html`)](../reports/enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-cards.html)

---

## 2. 📊 4-Gateway Architectural Comparison Matrix (`1,500 Peak RPS`, `20B Tokens/Mo`)

| Dimension | `Azure APIM (AI Gateway)` | `LiteLLM Proxy` | `Kong AI Gateway` | `Envoy AI Gateway / Higress` |
| :--- | :--- | :--- | :--- | :--- |
| **Core Runtime** | Managed Azure PaaS | Python (FastAPI) | Lua / Go / Rust Plugins | **C++ Envoy + K8s Gateway API** |
| **P99 Proxy Latency Tax** | `3.8 ms` | `11.4 ms` | `2.8 ms` | **`0.9 ms` ✓** |
| **429 / TTFT Failover Speed** | `210 ms` | `380 ms` | `190 ms` | **`120 ms` ✓** |
| **K8s `vLLM` KV/LoRA Routing** | Custom external callout | Basic prefix routing | Plugin-based | **Native `Inference Pool Extension` ✓** |
| **Provider SDK Translation** | Azure / Foundry / Bedrock | **Best (`100+` LLM providers)** | `20+` providers | OpenAI-compatible + major clouds |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Implement Two-Phase Streaming Weighted Token Rate Limiting (`Pre-Flight Reserve + Post-SSE Reconcile`)**.
- [ ] **Configure Sub-Second `TTFT (>1.5s)` and `HTTP 429` Spillover** across `PTU Primary -> PTU Secondary -> Self-Hosted K8s -> PAYGO`.
- [ ] **Enable K8s `Gateway API Inference Extension` Endpoint Picker** for self-hosted `vLLM / SGLang` pools.
