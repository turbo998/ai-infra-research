# Enterprise Cybersecurity AI Architecture: `Visual Quishing VLM Gate` & `Autonomous DevSecOps Kata Patching Sandbox` Sizing Guide

![Enterprise Cybersecurity Visual Phishing & DevSecOps Patching Summary](../../assets/svg/enterprise-cybersecurity-soc-phishing-vlm-devsecops-patch-sizing-faq-summary.svg)

> **Target Persona**: Cloud & AI Solution Architect / Enterprise Security Architect  
> **Core Scenarios**: Defending `50,000` employee mailboxes against zero-day QR/visual phishing (`Quishing`) and auto-remediating `15,000` monthly SAST/CVE code alerts across `2,000` repositories  
> **Interactive H5 Learning Workbench**: [`docs/h5/cybersecurity-soc-phishing-vlm-devsecops-sizer.html`](../h5/cybersecurity-soc-phishing-vlm-devsecops-sizer.html)

---

## 1. Executive Summary

1. **Zero-Day Visual & QR Phishing (`Quishing`) Defense (`56.0% -> 98.8%` Detection on `2× L40S`)**:
   - Text-only email filters miss `44%` of QR-code and HTML-as-image phishing lures.
   - Decoding QR URLs inside an isolated `Kata` headless browser, capturing the landing page screenshot, and comparing **`SigLIP / Qwen2.5-VL` visual brand identity against the actual `eTLD+1` certificate domain** blocks zero-day visual spoofing in `120ms` at **`98.8%` accuracy**.
2. **Closed-Loop DevSecOps Auto-Patching (`21 Days -> 14 Minutes` MTTR on `4× H200 + Kata MicroVMs`)**:
   - Filtering `35%` unreachable SAST false positives via AST dataflow analysis and verifying every generated patch inside a **`Kata MicroVM` (`100%` unit tests pass + `CodeQL/Semgrep` re-scan confirms `0` alerts)** before opening a PR lifts developer merge rate from **`28.0%` to `91.2%`**.
   - Total CISO fleet (`2× L40S + 4× H200` = **`$15,500/mo`**) saves **`89.3%`** vs. `$145,000/mo` commercial security add-ons and manual AppSec consulting.

---

## 2. Enterprise Security AI Comparison Matrix (`50K` Employees · `2,000` Repos)

| Dimension | Legacy Text SEG + Raw SAST Reports | Unverified LLM Auto-PR Bot | **Visual Brand VLM Gate (`2× L40S`) + Kata Verified Patch Loop (`4× H200`)** |
|---|---|---|---|
| **QR / Visual Phishing Block** | `56.0%` | `56.0%` | **`98.8%` (`+42.8 pts`)** |
| **Security PR Merge Rate** | Manual only | `28.0%` (`38%` break builds) | **`91.2%` (pre-verified against unit tests & CodeQL)** |
| **Critical CVE `MTTR`** | `21.0 days` | `4.5 days` | **`14.0 minutes`** |
| **Monthly Security Cost** | `$145,000 / mo` | `$48,000 / mo` | **`$15,500 / mo` (`-89.3%`)** |

---

## 3. Production Checklist

- [ ] Block any external link where **VLM-detected login brand identity (`Microsoft / Okta / SAP`) mismatches the verified `eTLD+1` domain**.
- [ ] Gate all AI-generated security PRs behind **isolated `Kata MicroVM` unit-test execution and SAST re-scan verification**.
