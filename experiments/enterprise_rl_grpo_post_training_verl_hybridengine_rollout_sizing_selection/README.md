# Enterprise Post-Training RL / GRPO Infrastructure (`veRL 3D-HybridEngine`, `FP8 Rollout` & Long-Tail Straggler Mitigation) (`#477`)

Reproducible analytical model comparing 4-Model PPO Disjoint Pools, Disaggregated `OpenRLHF` Ray+vLLM Pools, `veRL` Co-Located `3D-HybridEngine` (`1.8s` weight resharding), and `veRL/slime` + `FP8 Rollout` + Partial-Rollout Long-Tail Straggler Streaming (`86.4` GRPO steps/hr on 64 GPUs).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_rl_grpo_post_training_verl_hybridengine_rollout_sizing_selection/run.py --check`
