# Enterprise RL Post-Training & Reasoning Alignment Selection (`Azure AI Foundry RFT` vs. `veRL HybridEngine GRPO / RLVR` vs. `PPO / DPO`)

Reproducible analytical model comparing four enterprise post-training and reinforcement learning architectures (`SFT+DPO`, `4-Model PPO [36Ψ]`, `Azure AI Foundry RFT`, and `veRL HybridEngine GRPO/RLVR [18Ψ]`) across `VRAM footprint`, `Pass@1 accuracy`, `weight resharding latency`, and `training run cost`.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_rl_post_training_foundry_rft_vs_verl_grpo_dpo_aks_tco_selection/run.py`
