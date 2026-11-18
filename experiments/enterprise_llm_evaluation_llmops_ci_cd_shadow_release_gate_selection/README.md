# Enterprise LLM Evaluation, LLMOps CI/CD Release Gate & Shadow Replay Evaluator 

Deterministic analytical evaluator comparing four enterprise LLM evaluation and release-gating architectures across 25 AI applications (`120` releases/month, `500` golden cases/app):

1. **Tier 1 (`tier1_manual_spreadsheet_vibecheck`)**: Manual Spreadsheet Vibe-Check (`38.0%` regression catch rate, `120.0 hours` [5-day] release cycle, `$22,000/mo` human QA labor cost).
2. **Tier 2 (`tier2_uncalibrated_likert_llm_judge`)**: Uncalibrated 1–5 Likert Frontier LLM-as-a-Judge (`68.5%` regression catch rate, `0.56` Cohen's Kappa human agreement due to verbosity/position bias, `1.2 hours`, `$14,800/mo`).
3. **Tier 3 (`tier3_deterministic_plus_binary_rubric`)**: Deterministic Code Assertions + Decomposed Binary Rubric Judge (`Promptfoo / DeepEval / Ragas`, `91.2%` regression catch rate, `0.86` Cohen's Kappa, `0.37 hours` [`22 min`], `$2,900/mo`).
4. **Tier 4 (`tier4_cicd_gate_shadow_replay_flywheel`)**: CI/CD Golden Gate + Production Shadow Traffic Replay + Automated Bad-Case Flywheel (`98.6%` regression catch rate, `0.91` Cohen's Kappa, `0.30 hours` [`18 min`, `400x` faster than Tier 1], `$3,400/mo` [`-77.0%` vs. Tier 2]).

## Usage

```bash
python experiments/enterprise_llm_evaluation_llmops_ci_cd_shadow_release_gate_selection/run.py
python experiments/enterprise_llm_evaluation_llmops_ci_cd_shadow_release_gate_selection/run.py --check
```
