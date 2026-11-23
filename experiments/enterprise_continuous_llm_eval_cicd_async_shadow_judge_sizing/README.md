# Enterprise Continuous LLM Evaluation & Online Drift Governance: `Manual Spot-Check` vs. `100% Inline 70B Judge` vs. `CI/CD Golden Gate + 2ms Drift Sensor + 2% Async Kafka Shadow Judge` (`#539`)

- **Chinese FAQ**: [`../../docs/components/enterprise-continuous-llm-eval-cicd-async-shadow-judge-sizing-faq.md`](../../docs/components/enterprise-continuous-llm-eval-cicd-async-shadow-judge-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-continuous-llm-eval-cicd-async-shadow-judge-sizing-faq.en.md`](../../docs/components/enterprise-continuous-llm-eval-cicd-async-shadow-judge-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/continuous-llm-eval-cicd-async-shadow-judge-lab.html`](../../docs/h5/continuous-llm-eval-cicd-async-shadow-judge-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-continuous-llm-eval-cicd-async-shadow-judge-sizing-sizer.html`](../../docs/reports/enterprise-continuous-llm-eval-cicd-async-shadow-judge-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-continuous-llm-eval-cicd-async-shadow-judge-sizing-cards.html`](../../docs/reports/enterprise-continuous-llm-eval-cicd-async-shadow-judge-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_continuous_llm_eval_cicd_async_shadow_judge_sizing/run.py
python -m pytest tests/test_enterprise_continuous_llm_eval_cicd_async_shadow_judge_sizing.py -q
```
