# Enterprise SLM Knowledge Distillation & Synthetic Data Flywheel Evaluator (Round 392)

Deterministic analytical evaluator comparing four enterprise knowledge distillation and synthetic data flywheel pipelines for replacing a `$50,000/mo` 70B/671B Teacher (`86.0` baseline score) with an 8B Student SLM across `15,000,000` requests/month:

1. **Tier 1 (`tier1_manual_human_sme_annotation_sft`)**: Manual Human SME Annotation SFT (`67.5` score [`78.5%` teacher parity], `84 days` [12 weeks] lead time, `$65,000` upfront build cost, `$6,500/mo` serving cost).
2. **Tier 2 (`tier2_unfiltered_naive_teacher_synthetic_sft`)**: Unfiltered Naive Teacher Synthetic SFT (`72.1` score [`83.8%` parity due to teacher hallucination poisoning], `4 days` lead time, `$4,200` build cost, `$6,500/mo` serving cost).
3. **Tier 3 (`tier3_verified_rejection_sampling_sft`)**: Gateway Log Harvesting + Execution-Verified Rejection Sampling SFT (`Best-of-4` filtered by deterministic compiler/SQL/NLI verifiers, `81.0` score [`94.2%` parity], `3 days`, `$5,800` build cost, `$6,500/mo` serving cost).
4. **Tier 4 (`tier4_rejection_sft_dpo_plus_confidence_cascade_flywheel`)**: Verified Rejection SFT + Hard-Negative DPO/GRPO Alignment + `88/12` Logprob Confidence Cascade Flywheel (`85.5` score [`99.4%` teacher parity!], `3 days` lead time, `$6,400` build cost [`-90.2%` vs. Tier 1], `$9,200/mo` serving cost [`-81.6%` vs. `$50,000/mo` 100% Teacher]).

## Usage

```bash
python experiments/enterprise_slm_distillation_synthetic_data_flywheel_selection/run.py
python experiments/enterprise_slm_distillation_synthetic_data_flywheel_selection/run.py --check
```
