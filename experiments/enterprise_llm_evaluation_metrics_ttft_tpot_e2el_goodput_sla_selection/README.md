# Enterprise AI Evaluation Metrics (TTFT, TPOT/ITL, E2EL, Goodput & Effective $/1M Tokens): 4-Tier Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-llm-evaluation-metrics-ttft-tpot-e2el-goodput-sla-selection-faq.md`](../../docs/components/enterprise-llm-evaluation-metrics-ttft-tpot-e2el-goodput-sla-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-llm-evaluation-metrics-ttft-tpot-e2el-goodput-sla-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_llm_evaluation_metrics_ttft_tpot_e2el_goodput_sla_selection/run.py
python experiments/enterprise_llm_evaluation_metrics_ttft_tpot_e2el_goodput_sla_selection/run.py --check
python -m unittest tests/test_enterprise_llm_evaluation_metrics_ttft_tpot_e2el_goodput_sla_selection.py
```
