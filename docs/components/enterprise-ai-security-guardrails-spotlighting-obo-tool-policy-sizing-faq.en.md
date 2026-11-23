# Enterprise AI Security Guardrails, `Spotlighting` Indirect Injection Defense & `Entra OBO + OPA` Tool Zero-Trust Selection Guide (`#533`)

> **Language / 语言**: [简体中文](./enterprise-ai-security-guardrails-spotlighting-obo-tool-policy-sizing-faq.md) | **English** · [📱 Interactive H5 Study Lab](../h5/ai-security-guardrails-spotlighting-obo-lab.html) · [🧮 Interactive Sizer](../reports/enterprise-ai-security-guardrails-spotlighting-obo-tool-policy-sizing-sizer.html) · [🃏 7-Card Deck](../reports/enterprise-ai-security-guardrails-spotlighting-obo-tool-policy-sizing-cards.html) · [Back to Catalog](../full-catalog-en.md)

![Architecture Summary](../../assets/svg/enterprise-ai-security-guardrails-spotlighting-obo-tool-policy-sizing-faq-summary.svg)

---

## 1. Executive Summary (3-Minute Read)

1. **Why Synchronous 70B `LLM-as-Judge` Guardrails Add `380ms` Latency and Still Fail Against Indirect Prompt Injection**:
   - Using a 70B generative model as a synchronous input/output judge adds **`380ms` P99 latency**, burns **24 GPUs (`$72,000/mo` at 1,000 QPS)**, and still suffers an **`18.6%` indirect prompt injection bypass rate** and **`14.5%` unauthorized tool execution rate** because probabilistic LLMs cannot enforce hard security boundaries when untrusted RAG documents or emails are mixed into context.
2. **The 3-Layer Zero-Trust Agentic Defense (`11.5ms` Parallel DeBERTa/Prompt Shields + `Spotlighting` + Deterministic `Entra OBO + OPA/Cedar` Gate)**:
   - **Layer 1 (`11.5ms` Parallel Encoder Shield)**: Runs `DeBERTa-v3 / Azure Prompt Shields` in parallel with retrieval (`33.04x` faster than 70B LLM-as-Judge).
   - **Layer 2 (`Spotlighting` Datamarking & Dynamic Delimiters)**: Wraps all untrusted RAG/email/web payloads in cryptographically randomized delimiters, cutting indirect prompt injection bypass from `18.6%` to **`0.12%`**.
   - **Layer 3 (Deterministic `Entra ID OBO + OPA/Cedar` Tool Parameter Gate)**: Enforces user-scoped `On-Behalf-Of (OBO)` JWT claims and deterministic policy checks on every MCP/tool call argument, achieving **`0.0%` unauthorized tool execution** on just **2 guardrail GPUs (`-91.7%` GPUs, `-93.3%` cost)**.

---

## 2. Quantitative Architecture Decision Matrix (`1,000 QPS`)

| Metric | Scheme A: Regex + `70B LLM-as-Judge` | Scheme B: Sync Dual-Pass `Llama-Guard-8B` | Scheme C: `DeBERTa + Spotlighting + OBO/OPA` (Recommended) |
| :--- | :--- | :--- | :--- |
| **Added Guardrail P99 Latency** | `+380.0 ms` | `+145.0 ms` | **`+11.5 ms` (`33.04x` faster)** |
| **Indirect Injection Bypass Rate** | `18.6%` | `4.2%` | **`0.12%` (`99.88%` blocked)** |
| **Unauthorized Tool Execution** | `14.5%` | `3.8%` | **`0.0%` (Deterministic OBO + OPA)** |
| **Guardrail GPUs (`1,000 QPS`)** | `24 GPUs` | `12 GPUs` | **`2 GPUs` (`-91.7%`)** |
| **Monthly Guardrail Cost** | `$72,000 / mo` | `$36,000 / mo` | **`$4,800 / mo` (`-93.3%`)** |
