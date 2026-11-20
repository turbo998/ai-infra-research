# Enterprise AI PR Code Review, `SAST/SCA` Reachability & Compile-Verified Autofix Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-ai-pr-code-review-appsec-sast-reachability-autofix-selection-faq.md`](../../docs/components/enterprise-ai-pr-code-review-appsec-sast-reachability-autofix-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-ai-pr-code-review-appsec-sast-reachability-autofix-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_ai_pr_code_review_appsec_sast_reachability_autofix_selection/run.py
python experiments/enterprise_ai_pr_code_review_appsec_sast_reachability_autofix_selection/run.py --check
python -m unittest tests/test_enterprise_ai_pr_code_review_appsec_sast_reachability_autofix_selection.py
```
