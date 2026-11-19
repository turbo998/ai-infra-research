# Enterprise AI CI/CD, `EvalOps` Regression Gate & Canary Rollout Selection (`R485`)

Deterministic evaluation suite comparing four enterprise AI application CI/CD, prompt/model evaluation (`EvalOps`), and progressive release architectures across `120` monthly prompt/model releases (`5,000-case` golden evaluation dataset, `< 5.0 min` PR CI SLA):

1. **Tier 1**: 'Vibe-Check' Manual Prompt Edits & Blind Model Upgrades (`36.5%` production regression escape rate, `28.0 days` model migration lead time, `140.0 min` rollback MTTR, `$158,000/mo` incident + rework TCO)
2. **Tier 2**: Brute-Force `5,000-Case` Frontier LLM Judge on Every Git Pull Request (`6.8%` regression escape, `54.0 min` PR CI duration, `$132,000/mo` TCO)
3. **Tier 3**: Offline Golden-Set Eval Only with 100% Instant Cutover (`8.4%` regression escape on live distribution shift, `42.0 min` rollback MTTR, `$74,000/mo` TCO)
4. **Tier 4**: 2-Stage Tiered PR Eval Gate (`3.2 min`) + AI Gateway Dark-Shadow Traffic Replay + `5% -> 25% -> 100%` Auto-Rollback Canary (`0.4%` regression escape, `3.2 min` PR CI duration, `0.5 days = 12h` model migration lead time, `0.5 min = 30s` auto-rollback MTTR, `$26,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_ai_cicd_evalops_prompt_regression_canary_selection/run.py
python experiments/enterprise_ai_cicd_evalops_prompt_regression_canary_selection/run.py --check
python -m unittest tests/test_enterprise_ai_cicd_evalops_prompt_regression_canary_selection.py
```
