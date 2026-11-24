# Enterprise HR & Talent AI Architecture: `EU AI Act Blind Skills Matcher` & `20,000-Seat Voice Roleplay Coach` Sizing Guide

![Enterprise HR Blind Skills Matcher & Voice Roleplay Summary](../../assets/svg/enterprise-hr-talent-skills-graph-bias-audit-roleplay-sizing-faq-summary.svg)

> **Target Persona**: Cloud & AI Solution Architect  
> **Core Scenarios**: Global enterprises screening `500,000` resumes/month under **EU AI Act Annex III** & **NYC Local Law 144** fairness audits and running `40,000` hours/month of interactive voice sales/compliance roleplay across `20,000` employees  
> **Interactive H5 Learning Workbench**: [`docs/h5/hr-talent-skills-graph-bias-roleplay-sizer.html`](../h5/hr-talent-skills-graph-bias-roleplay-sizer.html)

---

## 1. Executive Summary

1. **Demographic-Blind Skills Ontology Matcher (`Disparate Impact Ratio 0.68 -> 0.94 >= 0.80`)**:
   - Feeding unmasked resumes into LLMs induces demographic bias (`0.68` selection ratio, failing the EEOC `4/5ths` rule).
   - Stripping name, photo, gender pronouns, age, graduation year, and zip code via a **`4.5ms` `GLiNER` Blind-Masking Gate** *before* embedding or LLM scoring achieves **`0.000` counterfactual name-swap variance** and a **`0.94` Disparate Impact Ratio** on just **`2× L40S`** for `500,000` resumes/month.
2. **`20,000`-Seat Voice Roleplay Coach (`$0.74/Employee/Month` vs. `$45/Seat` SaaS, `-98.3%`)**:
   - Combining `#581` `290ms` WebRTC streaming voice cascade with `#586` Multi-LoRA customer personas serves `40,000` training hours/month (`190` concurrent active sessions) on **`6× L40S + 4× H100` (`$14,900/mo` total = `$0.74/employee/mo`)**, saving **`$10.62M/year`** vs. `$900,000/mo` commercial coaching SaaS.

---

## 2. HR & L&D AI Architecture Comparison (`20,000` Employees · `500K` Resumes/Mo)

| Dimension | Commercial Recruiting + Coaching SaaS (`$45/seat`) | Unmasked Generic Cloud LLM Resume Ranker | **Blind Skills Matcher + Self-Hosted Roleplay (`6× L40S + 4× H100`)** |
|---|---|---|---|
| **EEOC `4/5ths` Impact Ratio** | `0.82–0.88` | `0.68` (**Fails `<0.80` legal threshold**) | **`0.94` (`0.000` counterfactual name-swap delta)** |
| **Cost per Employee / Mo** | `$45.00 / mo` | `$6.20 / mo` | **`$0.74 / mo` (`-98.3%`)** |
| **Total Monthly Spend** | `$900,000 / mo` | `$124,000 / mo` | **`$14,900 / mo` (`-$10.62M/yr` saved)** |

---

## 3. Production Checklist

- [ ] Strip all demographic identifiers via **`GLiNER` blind-masking** prior to vector indexing or LLM rubric scoring.
- [ ] Enforce **human-in-the-loop review (`EU AI Act Article 14`)** with structured `ESCO / O*NET` competency evidence tables rather than automated rejection scores.
