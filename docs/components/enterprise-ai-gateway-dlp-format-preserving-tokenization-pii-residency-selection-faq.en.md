# Enterprise AI Gateway Outbound Data Loss Prevention (`DLP`), Reversible Pseudonymization Vault (`NIST FF1-FPE` & Consistent Entity Mapping), Streaming SSE Re-Hydration & Cross-Border PII Residency Routing Architecture Selection Guide: Unfiltered Pass-Through Proxy vs. Destructive Static `<REDACTED>` Masking vs. Inline 7B LLM Scrubber + Full Buffering vs. `11ms INT8 ONNX NER` + Session-Consistent `FF1-FPE` Vault + `Aho-Corasick` Streaming Re-Hydrator & Sovereign Geo-Router

> **Language / 语言**: **English (Current)** | [中文](./enterprise-ai-gateway-dlp-format-preserving-tokenization-pii-residency-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Enterprise CISOs, Data Protection Officers (`GDPR / PIPL / HIPAA`), and AI Gateway (`Envoy AI Gateway / Kong / Azure APIM`) Engineers.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_ai_gateway_dlp_format_preserving_tokenization_pii_residency_selection/results.json`](../../experiments/enterprise_ai_gateway_dlp_format_preserving_tokenization_pii_residency_selection/results.json).

![Enterprise AI Gateway Outbound DLP, FF1-FPE Vault & Stream Re-Hydration Summary](../../assets/svg/enterprise-ai-gateway-dlp-format-preserving-tokenization-pii-residency-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating an enterprise AI Gateway governing **4.5 million daily prompts (`18.5%` containing customer PII, IBANs, payment cards, or cloud API secrets, and `35.0%` invoking `MCP` tool calls across `6` regulated jurisdictions)** exposes three critical architectural failure modes:
1. **Destructive Static `<REDACTED>` Masking (Tier 2) Breaks `31.4%` of Multi-Entity Reasoning and `MCP` Tool Calls**: When a prompt states "*Customer Alice (`DE89...3000`) claims her wire to Bob (`FR76...0189`) failed; invoke `refund_wire(sender, recipient)`*", replacing every entity with `<REDACTED>` collapses coreference ("*Customer `<REDACTED>` claims wire to `<REDACTED>` failed*") and causes the LLM to pass literal `"<REDACTED>"` into downstream banking APIs!
2. **Running a 7B LLM Inline for Scrubbing + Buffering the Full Response Before Re-Hydration (Tier 3) Adds `+640 ms` to P99 TTFT and Destroys Streaming UX**.
3. **Unfiltered Pass-Through Proxies Relying Solely on Contractual Zero-Data-Retention (`ZDR`, Tier 1) Leak `14.6%` of Pasted Developer Secrets/PII to External Clouds and Trigger `11.4%` Cross-Border `GDPR / PIPL` Residency Violations**.

Evaluated across **4.5M daily prompts, 18.5% PII/secret density, and 35% MCP tool-calling traffic**, the four AI Gateway DLP architectures compare as follows:

| Architectural Tier | Core Stack Combination | External Cloud PII/Secret Exposure | Coreference & `MCP` Tool Error | Added Gateway P99 Latency | Cross-Border Residency Violation | Monthly DLP & Rework TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Unfiltered Pass-Through (`ZDR Only`)** | Raw PII, IBANs & API keys sent directly to external cloud LLMs | `14.6%` ✕ *(raw PII & secret leak)* | `1.2%` | `2.0 ms` | `11.4%` ✕ *(illegal cross-border transfer)* | `$195,000 / mo` ✕ | ⛔ Violates banking/healthcare residency laws |
| **🟨 Tier 2: Destructive Static `<REDACTED>` Mask** | Replaces all names, accounts & emails with identical `<REDACTED>` | `3.8%` ✕ *(regex false negatives)* | `31.4%` ✕ *(entity collapse & broken tools)* | `8.5 ms` | `4.2%` ✕ | `$148,000 / mo` ✕ | ⛔ Breaks 1 in 3 multi-entity & tool workflows |
| **🟦 Tier 3: 7B LLM Scrubber + Full Stream Buffer** | Inline 7B LLM rewrite + buffering full 1,000-token output to replace | `0.6%` | `6.8%` ✕ | `640.0 ms` ✕ *(kills streaming TTFT)* | `0.9%` ✕ | `$112,000 / mo` ✕ | ⚠️ +640ms latency stall & heavy GPU overhead |
| **🟩 Tier 4: `11ms ONNX NER` + `FF1-FPE` Vault + SSE Re-Hydrator + Geo-Router** | **`INT8 DeBERTa-ONNX` + Checksum `AST` + Consistent `FF1-FPE` Vault + `12B` Sliding-Window SSE & `MCP` Re-Hydrator + Sovereign Router** | **`0.0%` 🏆** *(Zero leak)* | **`0.4%` 🏆** *(-31.0 pts)* | **`14.0 ms` 🏆** *(Streaming intact)* | **`0.0%` 🏆** *(Geo-pinned)* | **`$38,000 / mo` 🏆** *(-80.5%)* | 🌟 **Recommended Enterprise AI Gateway Standard** |

> 💡 **Golden Rule for Enterprise AI Gateway & Security Architects**: **"Never scrub prompts with destructive `<REDACTED>` strings that blind multi-entity reasoning and break tool arguments—intercept sensitive spans in `<14ms` using `SIMD` checksum/entropy filters + `INT8 DeBERTa-ONNX NER`, map each unique entity deterministically into an ephemeral session vault (`Alice -> [PER_01]`, `Bob -> [PER_02]`, `IBAN -> NIST FF1-FPE` format-preserving surrogate), and re-hydrate real values on the fly across both SSE response streams (via a 12-byte lookahead trie) and outbound `MCP` tool-call arguments."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 AI Gateway DLP & Privacy Pain Points)

### Q1: Why does `<REDACTED>` masking break `31.4%` of Agent tool calls, and how does a Session-Consistent `NIST FF1-FPE` Vault achieve `99.6%` accuracy with `0.0%` cloud leakage?
- **First-Principles Reality**:
  - Collapsing distinct entities into an identical `<REDACTED>` token destroys relational graph structure ($A \ne B$) and causes the LLM to emit `"<REDACTED>"` inside JSON tool arguments.
  - Tier 4 maintains an ephemeral, session-scoped **Bijective Redis Vault** (`TTL = 30 min`) that assigns consistent typed tokens (`[PER_01]`, `[PER_02]`) and **NIST `FF1-FPE` (Format-Preserving Encryption)** surrogates (`DE89...3000` $\leftrightarrow$ `DE00111122223333444401`).
  - The cloud LLM reasons over valid, distinct surrogates and emits `refund_wire(sender="DE00...01", recipient="FR00...02")`. The gateway intercepts the `MCP` tool call and **re-hydrates the real IBANs in `<1.5ms` before invoking the internal banking API** (`0.4%` tool error, `0.0%` cloud exposure—even immune to prompt-injection exfiltration!).

### Q2: How does the 12-Byte `Aho-Corasick` Lookahead Stream Buffer re-hydrate split SSE tokens (`[PER_` + `01]`) without adding `+640 ms` of full-response buffering?
- **Subword-Split-Safe Streaming Automaton**:
  - When an LLM streams SSE tokens, a placeholder like `[PER_01]` is frequently split across two network chunks (`Chunk #14: "...to [PER_"`, `Chunk #15: "01] now"`).
  - Instead of buffering the entire 1,000-token response (Tier 3, `+640ms`), Tier 4 runs a **Prefix-Trie Streaming Automaton** that passes normal tokens through with zero delay and holds at most **12 bytes (`1–2 token steps`)** only when an unclosed bracket prefix `[` is detected at a chunk boundary—completing the replacement the instant `]` arrives (`14.0 ms` total P99 gateway latency).

### Q3: How do `SIMD` Checksum/Entropy Filters and Geo-Residency Routers prevent false positives on order IDs and enforce `GDPR / PIPL` data sovereignty?
- **Luhn / `ISO 7064 MOD-97` Verification + Jurisdiction-Pinned Routing**:
  - **1.2ms SIMD Pre-Filter**: Verifies Luhn check digits on credit cards, `MOD-97-10` check digits on IBANs, and Shannon entropy (`>4.2 bits/char`) on `AWS / GitHub / JWT` secrets before invoking the `11ms INT8 DeBERTa-ONNX` NER model.
  - **Sovereign Geo-Router**: Requests tagged with `GDPR Art. 9 Special Category` or `PIPL Restricted` metadata are blocked from cross-border egress and routed strictly to in-jurisdiction Confidential GPU TEEs (`0.0%` residency violation).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + FF1-FPE Vault Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise AI Gateway Outbound DLP, Tool Re-Hydration & Residency<br/>(4.5M Prompts/Day + 18.5% PII/Secrets + 35% MCP Tool Calls + 6 Jurisdictions)"] --> B{"Uses unfiltered pass-through proxy (Tier 1)<br/>or destructive static <REDACTED> masking (Tier 2)?"}
    B -- "Yes (Pass-through or static <REDACTED>)" --> C["⛔ 14.6% PII Leakage or 31.4% Broken Tool Calls (Tier 1 / Tier 2)!<br/>Collapses multi-entity identity & passes literal '<REDACTED>' to APIs"]
    B -- "Uses reversible pseudonymization" --> D{"Combines 11ms INT8 ONNX NER + consistent FF1-FPE vault<br/>with 12B sliding-window SSE & MCP tool-boundary re-hydration?"}
    D -- "No (Inline 7B LLM scrubber + full response buffer)" --> E["⚠️ +640ms P99 TTFT Stall & High GPU Cost (Tier 3)<br/>Full-stream buffering breaks real-time SSE typing UX"]
    D -- "Yes (ONNX NER + FF1-FPE Vault + Stream/MCP Re-Hydrator)" --> F["🟩 Tier 4: Consistent FF1-FPE Vault + SSE/MCP Re-Hydrator + Geo-Router 🏆<br/>(0.0% Cloud PII Leak | 0.4% Tool Error | 14.0 ms P99 | -80.5% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Consistent `FF1-FPE` Session Vault + Sliding-Window SSE & `MCP` Re-Hydration Topology

```mermaid
flowchart LR
    subgraph Inbound["1. 14ms Inbound Detection & Pseudonymization"]
        Req["User Prompt / Stack Trace<br/>(Alice, Bob, IBAN, AWS Key)"] --> Detect["⚡ 1.2ms Luhn/MOD-97/Entropy AST<br/>+ 11ms INT8 DeBERTa-ONNX NER"]
        Detect --> Vault[("🔐 Ephemeral Session Vault (Redis)<br/>Alice ↔ [PER_01] | Bob ↔ [PER_02]<br/>IBAN ↔ NIST FF1-FPE Surrogate")]
    end

    subgraph Cloud["2. Sovereign Geo-Router & LLM Pool"]
        Vault -->|Surrogate-Only Prompt| Geo{"🌍 Jurisdiction Geo-Router<br/>(GDPR / PIPL / HIPAA)"}
        Geo -->|Standard Pseudonymized| Ext["☁️ External Cloud LLM<br/>(0.0% Raw PII Exposure)"]
        Geo -->|Restricted Sovereign| Sovereign["🛡️ In-Country Confidential GPU TEE"]
    end

    subgraph Rehydrate["3. Dual-Path Real-Time Re-Hydration"]
        Ext & Sovereign -->|SSE Token Stream| Trie["🔄 12-Byte Lookahead SSE Re-Hydrator<br/>Joins Split [PER_ + 01] -> Alice"]
        Ext & Sovereign -->|MCP tool_calls| MCP["🔧 MCP Tool-Boundary Re-Hydrator<br/>Restores Real IBANs Before API Call"]
        Vault -.-> Trie & MCP
    end
```

### 3.3 End-to-End Sequence: Pseudonymizing a Dual-IBAN Wire Dispute & Re-Hydrating `MCP` Tool Arguments

```mermaid
sequenceDiagram
    autonumber
    participant User as Enterprise Agent Client
    participant GW as AI Gateway (ONNX NER + FF1-FPE Vault)
    participant LLM as External Cloud LLM (Zero Raw PII)
    participant Bank as Internal Core Banking MCP Server

    User->>GW: Prompt: "Alice Vance (DE89..3000) claims wire to Bob Miller (FR76..0189) failed; call refund_wire"
    GW->>GW: Detect 2 names + 2 MOD-97 IBANs in 14ms -> store in Session Vault as [PER_01], [PER_02], DE00..01, FR00..02
    GW->>LLM: Forward sanitized prompt: "[PER_01] (DE00..01) claims wire to [PER_02] (FR00..02) failed..." (0.0% cloud leak!)
    LLM-->>GW: Emit MCP tool call: refund_wire(sender="DE00..01", recipient="FR00..02")
    GW->>GW: ⚡ Intercept tool_call at gateway boundary -> swap surrogates back to real DE89..3000 & FR76..0189!
    GW->>Bank: Execute internal MCP call refund_wire("DE89..3000", "FR76..0189") -> return confirmation #TX-998
    LLM-->>GW: Stream SSE response (`Chunk #14: "...for [PER_"`, `Chunk #15: "01] (#TX-998)"`)
    GW-->>User: 12-byte lookahead buffer joins split token & emits: "...for Alice Vance (#TX-998)"!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Pass-Through (`ZDR Only`) | 🟨 Tier 2: Static `<REDACTED>` Mask | 🟦 Tier 3: 7B Scrubber + Full Buffer | 🟩 Tier 4: `FF1-FPE` Vault + SSE/MCP Re-Hydrator |
|---|---|---|---|---|
| **External Cloud PII/Secret Exposure** | `14.6%` ✕ (raw PII & keys leaked) | `3.8%` ✕ (regex misses context PII) | `0.6%` | 🏆 **`0.0%` (checksum/entropy `AST` + `INT8 ONNX NER`)** |
| **Coreference & `MCP` Tool Error** | `1.2%` | `31.4%` ✕ (entity collapse & broken args) | `6.8%` ✕ (LLM rewrite drift) | 🏆 **`0.4%` (consistent `[PER_k]` + `FF1-FPE` + `MCP` swap)** |
| **Gateway Added P99 Latency** | `2.0 ms` | `8.5 ms` | `640.0 ms` ✕ (full-stream buffer stall) | 🏆 **`14.0 ms` (`11ms ONNX NER` + `12B` lookahead trie)** |
| **Cross-Border Residency Violation** | `11.4%` ✕ (unfiltered cross-border) | `4.2%` ✕ | `0.9%` ✕ | 🏆 **`0.0%` (jurisdiction classification geo-router)** |
| **Monthly DLP & Rework TCO** | `$195,000 / mo` | `$148,000 / mo` | `$112,000 / mo` | **`$38,000 / mo` 🏆 (-80.5%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise AI Gateway Outbound DLP, FF1-FPE Vault & Residency Sizer](../reports/enterprise-ai-gateway-dlp-format-preserving-tokenization-pii-residency-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-ai-gateway-dlp-format-preserving-tokenization-pii-residency-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_ai_gateway_dlp_format_preserving_tokenization_pii_residency_selection/run.py --check
  python -m unittest tests/test_enterprise_ai_gateway_dlp_format_preserving_tokenization_pii_residency_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Ban Destructive Static `<REDACTED>` Replacement on Multi-Entity and Tool-Calling Prompts**: Use session-consistent indexed placeholders (`[PER_01]`, `[PER_02]`) and format-preserving encryption (`NIST FF1-FPE`) so the LLM preserves relational coreference.
- [ ] **Re-Hydrate Surrogate Tokens at Both the Outbound `MCP` Tool-Call Boundary and Inbound SSE Response Stream**: Ensure internal enterprise APIs receive real account/order IDs while external cloud LLMs see only synthetic surrogates.
- [ ] **Use a 12-Byte Prefix-Trie Lookahead Buffer on SSE Streams Instead of Full-Response Buffering**: Re-hydrate placeholders split across subword chunks (`[PER_` + `01]`) with `<15ms` total P99 latency overhead.
- [ ] **Run `1.2ms SIMD` Checksum (`Luhn / MOD-97`) and Shannon Entropy (`>4.2 bits/char`) Filters Ahead of `INT8 DeBERTa-ONNX` NER**: Catch API keys, JWTs, credit cards, and IBANs deterministically without false-flagging normal timestamps.
- [ ] **Enforce Jurisdiction Data-Residency Routing (`GDPR / PIPL / HIPAA`) at the AI Gateway**: Pin restricted special-category workloads to in-country Confidential GPU TEEs or on-prem clusters.
