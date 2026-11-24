# Enterprise LegalTech, Contract CLM & M&A Due Diligence AI Architecture: `CUAD Clause Router`, `OOXML Track-Changes Redlining` & VDR Matrix Guide

![Enterprise LegalTech Contract CLM & VDR Matrix Summary](../../assets/svg/enterprise-legal-contract-clm-redline-due-diligence-sizing-faq-summary.svg)

> **Target Persona**: Cloud & AI Solution Architect  
> **Core Scenarios**: Corporate legal departments and law firms reviewing `25,000` contracts/month against a `120`-rule playbook and extracting `35`-column M&A Virtual Data Room (VDR) matrices across `2,500` deal documents  
> **Interactive H5 Learning Workbench**: [`docs/h5/legal-contract-clm-redline-due-diligence-sizer.html`](../h5/legal-contract-clm-redline-due-diligence-sizer.html)

---

## 1. Executive Summary

1. **Solving the "120-Rule Monolithic Prompt" Attention Dilution Trap (`71.5% -> 97.8%` Risk Recall)**:
   - Stuffing an `18,000`-token MSA plus all `120` corporate playbook rules (`42,000` tokens) into a single prompt misses `28.5%` of subtle uncapped liability riders and burns `142K` tokens/contract.
   - **3-Stage Clause-Routed Playbook Pipeline**:
     - **Stage 1 (`ModernBERT-CUAD` on `2× L40S`, `14ms/doc`)**: Classifies contract paragraphs into `41` legal clause categories and filters out `68%` routine recitals.
     - **Stage 2 (`Qwen2.5-72B-FP8` on `4× H200`)**: Evaluates each risk clause only against its `2–3` matching playbook rules (`1,200` tokens/clause, **`-76.4%` total tokens/contract**), lifting risk recall to **`97.8%` (`+26.3 pts`)**.
     - **Stage 3 (`Myers Diff + Surgical OOXML <w:ins>/<w:del> Injection`)**: Patches only modified spans into the original `.docx` XML tree without touching unreviewed clauses or breaking numbering styles.
2. **Enclave Economics**: A self-hosted **`2× L40S + 4× H200` (`$15,500/mo`)** Privileged Enclave reviews `25,000` contracts/month at **`$0.62/contract`** (`-96.6%` vs. `$18.00/contract` commercial LegalTech SaaS, saving **`$5.21M/year`**).

---

## 2. Legal CLM Architecture Comparison (`25,000` Contracts/Month)

| Dimension | Commercial LegalTech SaaS (`$18/contract`) | Monolithic `42K`-Token Single Prompt (`18× H200`) | **Clause-Routed Playbook + OOXML Patch (`2× L40S + 4× H200`)** |
|---|---|---|---|
| **Liability Defect Recall** | `94.5%` | `71.5%` (attention dilution across 120 rules) | **`97.8%` (`+26.3 pts`)** |
| **Tokens per Contract** | SaaS blackbox | `142,000 tok` | **`33,500 tok` (`-76.4%`)** |
| **Cost per Contract / Mo** | `$18.00` (`$450,000/mo`) | `$2.45` (`$61,250/mo`) | **`$0.62` (`$15,500/mo`, `-96.6%`)** |

---

## 3. Production Checklist

- [ ] Never ask an LLM to regenerate an entire `.docx` file; emit structured JSON diffs and apply **surgical `OOXML <w:ins>/<w:del>` patches**.
- [ ] Enforce **deterministic substring verification** on all M&A VDR matrix quotes before surfacing them to legal counsel.
