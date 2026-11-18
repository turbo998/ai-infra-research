# Enterprise Reasoning Model Thinking Budget & Speculative Decoding Evaluator 

Deterministic analytical evaluator comparing four enterprise serving architectures for Reasoning Models (`DeepSeek-R1`, `o3-mini`, `Qwen-QwQ-32B`, `Claude Extended Thinking`) across `10,000,000` queries/month:

1. **Tier 1 (`tier1_unbounded_full_reasoning_always`)**: 100% Unbounded Deep Reasoning Mode (`3,250` avg `<think>` tokens/req, `93.5%` accuracy, `24.50 s` avg E2E latency, `$82,000/mo`).
2. **Tier 2 (`tier2_naive_hard_max_tokens_cap`)**: Naive Hard `max_tokens=1024` Cap (`890` avg `<think>` tokens, `18.4%` truncated mid-thought failure rate with empty final answer, `74.0%` accuracy, `$24,800/mo`).
3. **Tier 3 (`tier3_three_gear_reasoning_effort_router`)**: 3-Gear Difficulty Router (`70%` Non-Thinking Fast Mode + `20%` Low-Think + `10%` High-Think, `680` avg `<think>` tokens, `93.2%` accuracy, `4.20 s` avg latency, `$21,500/mo`).
4. **Tier 4 (`tier4_adaptive_budget_forcing_plus_mtp_speculative`)**: 3-Gear Router + Graceful `</think>` Budget-Forcing Stop + Native MTP / EAGLE-3 Speculative Decoding (`620` avg `<think>` tokens [`-80.9%` vs. Tier 1], `94.6%` accuracy, `0.0%` mid-thought truncation, `1.85 s` avg E2E latency [`13.24x` faster than Tier 1], `$12,800/mo` [`-84.4%` cost savings]).

## Usage

```bash
python experiments/enterprise_reasoning_model_thinking_budget_speculative_selection/run.py
python experiments/enterprise_reasoning_model_thinking_budget_speculative_selection/run.py --check
```
