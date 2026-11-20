# Enterprise AI FinOps Showback/Chargeback, PTU Amortization & Dynamic Budget Downgrade Selection (`R494`)

Deterministic quantitative evaluation model comparing four Enterprise AI FinOps and multi-tenant token governance architectures across **100B monthly enterprise tokens and 45 business units**:
- **Tier 1**: Unmetered Shared Central IT API Key ("Free Buffet" Anti-Pattern)
- **Tier 2**: Siloed Per-Department Subscriptions + Hard 100% Cutoff (`HTTP 429` Cliff)
- **Tier 3**: Flat Token-Count Showback Only (Ignoring `Input vs. Cached vs. Output` & `Peak vs. Off-Peak` Weights)
- **Tier 4**: `FOCUS 1.0` Compute-Weighted Chargeback (`0.15x Cache / 1x In / 4x Out` + `0.25x Night PTU`) + `80/95/100%` Graceful Model Downgrade + Runaway-Loop Breaker

## Verification

```bash
python experiments/enterprise_ai_finops_showback_chargeback_token_budget_governance_selection/run.py --check
python -m unittest tests/test_enterprise_ai_finops_showback_chargeback_token_budget_governance_selection.py
```
