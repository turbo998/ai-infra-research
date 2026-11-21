# LLM Reinforcement Learning Post-Training (`RLHF / PPO / GRPO / RLVR`) Architecture Selection: `36Ψ PPO` vs. `18Ψ GRPO`, Colocated `veRL HybridEngine` vs. Disaggregated `Slime`, `1.6s` Weight Sync & Partial Rollout Guide

> **Language**: [中文](./llm-rl-post-training-grpo-ppo-verl-slime-weight-sync-rollout-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and LLM Post-Training / RL Infrastructure Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![LLM RL Post-Training PPO vs GRPO, veRL & Partial Rollout Summary](../../assets/svg/llm-rl-post-training-grpo-ppo-verl-slime-weight-sync-rollout-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **In reasoning reinforcement learning (`GRPO / RLVR`), `80%+` of wall-clock step time is spent in autoregressive `Rollout Generation` (Inference)! Optimize RL post-training across three layers: (1) replace 4-model `PPO` (`36Ψ` static HBM) with critic-free `GRPO + Rule Verifiers` (`18Ψ` static HBM, saving `50%` memory); (2) replace disk-based weight syncing (`95s`) with in-memory zero-copy resharding (`veRL 3D-HybridEngine / RDMA P2P`) that casts `BF16` training shards directly into `FP8 vLLM/SGLang` rollout weights in **`1.6 seconds`**; and (3) eliminate the `32K` long-tail rollout barrier bubble (`38.5%–64.2%` idle GPUs) via `Partial Rollout Continuation` (pausing unfinished long chains and resuming their KV Cache in step $t+1$ with bounded staleness $k=1$) plus `Dynamic Sequence Packing`—cutting RL step duration `6.69x` (`495s -> 74s`) and reducing cost per 1,000 RL steps by `85.0%` (`$9,300`)!**

- **Interactive RL Memory & Step Sizer**: [Open Interactive Sizer (`llm-rl-post-training-grpo-ppo-verl-slime-weight-sync-rollout-sizer.html`)](../reports/llm-rl-post-training-grpo-ppo-verl-slime-weight-sync-rollout-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`llm-rl-post-training-grpo-ppo-verl-slime-weight-sync-rollout-selection-cards.html`)](../reports/llm-rl-post-training-grpo-ppo-verl-slime-weight-sync-rollout-selection-cards.html)

---

## 2. 📊 4-Tier RL Post-Training Comparison Matrix (`128x H200`, `32K` Max Rollout)

| Metric | Tier 1: 4-Model `PPO (36Ψ)` + Disk Weight Sync | Tier 2: `GRPO (18Ψ)` + Sync Colocated `veRL` | Tier 3: Static 50/50 Disaggregated Pools | Tier 4 (Recommended): `GRPO` + `FP8 Rollout` + `Partial Rollout Continuation` |
| :--- | :--- | :--- | :--- | :--- |
| **RL Step Wall-Clock Duration** | `495.0 s` ✕ | `198.0 s` | `162.0 s` | **`74.0 s` ✓ (`6.69x` faster)** |
| **Actor -> Rollout Weight Sync** | `95.0 s` ✕ | `1.8 s` ✓ | `4.2 s` | **`1.6 s` ✓ (In-Memory `FP8`)** |
| **Long-Tail Rollout Idle Bubble** | `64.2%` ✕ | `38.5%` ✕ | `26.4%` | **`5.8%` ✓ (Partial KV Resume)** |
| **Cost per 1,000 RL Steps** | `$62,000` | `$24,800` | `$20,300` | **`$9,300` ✓ (`-85.0%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Adopt Critic-Free `GRPO (18Ψ)` Over 4-Model `PPO (36Ψ)`** for verifiable math, coding, and tool-use RL.
- [ ] **Execute Rollouts in `FP8` on `vLLM / SGLang` with `<2s` In-Memory Weight Resharding**.
- [ ] **Enable `Partial Rollout Continuation` & `Variable-Length Sequence Packing`** to eliminate `32K–64K` long-tail barriers.
