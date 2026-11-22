# Enterprise LLM Overload Protection, Predictive TTFT/KV Admission Control & 4-Stage Graceful Degradation Model (`#459`)

Reproducible analytical model comparing unbounded FIFO queueing with retry storms, static HTTP QPS rate limiting, predictive TTFT/KV admission control + abort propagation, and 4-Stage Graceful Degradation (`80%` MTP cutoff $\to$ `85%` `<think>` cap $\to$ `90%` Cloud/SLM spillover $\to$ `95%` priority shedding).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_llm_overload_admission_control_kv_watermark_graceful_degradation_selection/run.py --check`
