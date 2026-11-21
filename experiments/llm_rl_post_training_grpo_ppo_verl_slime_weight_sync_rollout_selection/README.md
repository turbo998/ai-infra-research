# LLM RL Post-Training Architecture (PPO vs. GRPO, Colocated veRL HybridEngine vs. Disaggregated Async Rollout) Benchmark

Deterministic quantitative evaluation model for [`docs/components/llm-rl-post-training-grpo-ppo-verl-slime-weight-sync-rollout-selection-faq.md`](../../docs/components/llm-rl-post-training-grpo-ppo-verl-slime-weight-sync-rollout-selection-faq.md) and [`.en.md`](../../docs/components/llm-rl-post-training-grpo-ppo-verl-slime-weight-sync-rollout-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_rl_post_training_grpo_ppo_verl_slime_weight_sync_rollout_selection/run.py
python experiments/llm_rl_post_training_grpo_ppo_verl_slime_weight_sync_rollout_selection/run.py --check
python -m unittest tests/test_llm_rl_post_training_grpo_ppo_verl_slime_weight_sync_rollout_selection.py
```
