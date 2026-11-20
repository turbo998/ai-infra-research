# Enterprise AI Progressive Delivery, Session-Sticky Canary & Shadow Mirroring Selection Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-selection-faq.md`](../../docs/components/enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection/run.py
python experiments/enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection/run.py --check
python -m unittest tests/test_enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection.py
```
