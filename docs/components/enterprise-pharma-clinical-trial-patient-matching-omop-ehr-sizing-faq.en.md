# Enterprise Pharma Clinical Trial Patient-Protocol Matching (`10M EHR Patients`): `Brute-Force 70B` vs. `ICD-10 SQL Only` vs. `OMOP SQL Filter (94%) + 8B Negation-Aware EHR Feature Store + 70B Verifier` (`#547`)

> **Language / 语言**: [简体中文](./enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-faq.md) | **English** · [📱 Interactive H5 Study Lab](../h5/pharma-clinical-trial-patient-matching-omop-ehr-lab.html) · [🧮 Interactive Sizer](../reports/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-sizer.html) · [🃏 7-Card Deck](../reports/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-cards.html) · [Back to Catalog](../full-catalog-en.md)

![Architecture Summary](../../assets/svg/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-faq-summary.svg)

---

## 1. Executive Summary (3-Minute Read)

1. **Why Brute-Force `70B LLM` Scanning Across 10M Raw EHR Records Requires 480 GPUs (`$1.44M/mo`) with `76.4%` Precision While `ICD-10 SQL` Misses `62.0%` of Eligible Patients**:
   - Oncology and rare-disease protocols depend on biomarkers (`EGFR L858R`, `PD-L1 CPS`), `ECOG` status, and prior lines of therapy buried in unstructured clinical notes (`38.0%` recall with SQL-only).
   - Scanning all raw EHR notes per trial with a `70B LLM` confuses negated findings (`"no evidence of brain metastases"`) and family history, yielding **`76.4%` precision** on **480 H200 GPUs**.
2. **The 3-Stage `OMOP CDM SQL Filter (94% Pruned) + Incremental 8B Negation-Aware Patient Feature Store + 70B Criterion Verifier` Architecture**:
   - Prunes `94.0%` of ineligible patients in `15ms` via structured `OMOP SQL` (age, labs, ICD), maintains a reusable **Patient-Centric Feature Store** extracted once per visit by an `8B` clinical model with explicit `assertion_status` (`PRESENT | NEGATED | FAMILY_HISTORY`), and runs a `70B` criterion-by-criterion verifier on the top `6%` candidates — achieving **`96.8%` recall (`+58.8 pts`)** and **`98.2%` precision (`+21.8 pts`)** on just **12x H200 GPUs (`-97.5%` GPUs)**!

---

## 2. Quantitative Architecture Decision Matrix (`10M EHR Patients`)

| Metric | Scheme A: Brute-Force `70B` on Raw EHR Notes | Scheme B: Structured `ICD-10 SQL` Only | Scheme C: `OMOP SQL (94%) + 8B Feature Store + 70B Verifier` |
| :--- | :--- | :--- | :--- |
| **Trial Eligibility Recall** | `91.5%` | `38.0%` | **`96.8%` (`+58.8 pts` vs. SQL-Only)** |
| **Inclusion/Exclusion Precision** | `76.4%` | `89.0%` | **`98.2%` (`+21.8 pts` vs. Brute 70B)** |
| **Required H200 GPUs** | `480 GPUs` | `0 GPUs` | **`12 GPUs` (`-97.5%` GPUs)** |
| **Monthly Infra Cost** | `$1,440,000 / mo` | `$15,000 / mo` | **`$36,000 / mo` (`-97.5%` vs. Brute 70B)** |
