# Enterprise Financial Real-Time Fraud & Risk Control (`5,000 TPS · P99 <= 35ms`) 3-Tier Fast/Slow AI Sizer (`#519`)

Reproducible analytical model comparing **100% Inline 70B LLM (`680ms` P99, `320 GPUs` ✕)**, **Legacy XGBoost/Rules Only (`78.5%` recall, `6.8%` false positives ✕)**, and **3-Tier Fast/Slow AI Cascade (`92%` 3.2ms ONNX GBDT + `7.5%` 21.5ms 8B Single-Token Logprob Classifier + `0.5%` Async 70B/R1 SAR Investigation, `24.8ms` P99, `96.8%` recall, `12x H200 GPUs`, `-96.3%` GPUs 🏆)**.

## Run

```bash
python experiments/enterprise_finserv_realtime_fraud_risk_gbdt_slm_llm_cascade_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-finserv-realtime-fraud-risk-gbdt-slm-llm-cascade-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-finserv-realtime-fraud-risk-gbdt-slm-llm-cascade-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/finserv-realtime-fraud-risk-gbdt-slm-llm-cascade-lab.html](../../docs/h5/finserv-realtime-fraud-risk-gbdt-slm-llm-cascade-lab.html)
