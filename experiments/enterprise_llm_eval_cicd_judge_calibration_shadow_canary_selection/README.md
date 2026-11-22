# Milestone `#505`: Enterprise LLM Evaluation, CI/CD Regression Gate, `LLM-as-a-Judge` Calibration & Shadow Canary Selection

Reproducible analytical model comparing 4 enterprise LLM/Agent evaluation and release-gating architectures:
- **Tier 1 (`Manual 20-Prompt Vibe Check`)**: `28.0%` regression catch rate, `240 min` human review delay (`$800/release`).
- **Tier 2 (`Brute-Force 50,000-Prompt Cloud Judge with Uncalibrated 1-10 Scale`)**: `84.0%` catch rate, `0.51` Cohen's Kappa human agreement (position & verbosity bias), `390 min (6.5h)` CI delay (`$1,250/release`).
- **Tier 3 (`Public Academic Benchmarks Only: MMLU / GSM8K`)**: `35.0%` enterprise business regression catch rate.
- **Tier 4 (`Stratified 200-Hard-Case PR Gate + Position-Swapped Binary-Rubric 14B Judge + 1% Production Shadow Canary`)**: Catches **`98.4%` of regressions**, raises human expert agreement to **`0.89` Cohen's Kappa**, completes PR gating in **`2.8 minutes` (`139.3x` faster)**, and costs just **`$3.40 / release` (`-99.7%` cost)**.

## Run
```bash
python experiments/enterprise_llm_eval_cicd_judge_calibration_shadow_canary_selection/run.py
```
