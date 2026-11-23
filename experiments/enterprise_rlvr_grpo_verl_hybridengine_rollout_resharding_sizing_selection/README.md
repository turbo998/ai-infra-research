# Enterprise `RLVR / GRPO` Post-Training (`veRL HybridEngine / slime` Weight Resharding) Sizer (`#523`)

Reproducible analytical model comparing **Static Disaggregated Actor/Learner Pools (`54.0%` GPU idle bubble, `18.5s` weight sync, `64 GPUs`)**, **Synchronous Colocated Engine without Long-Tail CoT Pruning (`31.5%` bubble)**, and **`veRL HybridEngine / slime` In-Memory 3D Weight Resharding (`0.9s` sync) + Partial Rollout Straggler Pruning (`8.4%` bubble, `2.45x` step throughput, `24x H200 GPUs`, `-62.5%` GPUs 🏆)**.

## Run

```bash
python experiments/enterprise_rlvr_grpo_verl_hybridengine_rollout_resharding_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-rlvr-grpo-verl-hybridengine-rollout-resharding-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-rlvr-grpo-verl-hybridengine-rollout-resharding-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/rlvr-grpo-verl-hybridengine-rollout-resharding-lab.html](../../docs/h5/rlvr-grpo-verl-hybridengine-rollout-resharding-lab.html)
