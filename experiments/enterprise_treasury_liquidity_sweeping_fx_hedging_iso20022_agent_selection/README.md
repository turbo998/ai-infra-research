# Enterprise Global Corporate Treasury, Liquidity Sweeping, `ISO 20022` Payment Factory & FX `CVaR` Agent (`R443`)

Deterministic evaluation suite comparing four multinational corporate treasury, cash-flow forecasting, multilateral In-House Bank (`IHB`) netting, and FX hedging architectures across `45` global subsidiaries (`180` bank accounts, `$450M/month` cross-border/intercompany flows, `18` currencies):

1. **Tier 1**: Manual Subsidiary Excel Cash Reports + Gross Bilateral Wires + Spot FX (`22.5%` idle cash drag, `19.4%` 30-day cash MAPE, `42.0 bps` FX/wire cost, `6.8%` intraday overdraft breach, `$680,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM FX Directional Trader & Cash Summarizer (`21.0%` idle cash drag, `23.8%` MAPE, `39.5 bps` FX cost, `8.4%` overdraft breach, `$795,000/mo` loss)
3. **Tier 3**: Aggregate Time-Series ARIMA Cash Forecast (`11.4%` idle cash drag, `9.8%` MAPE, `24.0 bps` FX cost, `2.9%` overdraft breach, `$310,000/mo` loss)
4. **Tier 4**: Invoice-Survival Cash Engine + Multilateral `IHB` Netting Matrix + `ISO 20022 ZBA` Sweep + Regulatory Capital-Control `AST` + `CVaR` FX Agent (`2.2%` idle cash drag, `2.6%` 30-day MAPE, `3.5 bps` FX/wire cost, `0.0%` overdraft breach, `$56,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_treasury_liquidity_sweeping_fx_hedging_iso20022_agent_selection/run.py
python experiments/enterprise_treasury_liquidity_sweeping_fx_hedging_iso20022_agent_selection/run.py --check
python -m unittest tests/test_enterprise_treasury_liquidity_sweeping_fx_hedging_iso20022_agent_selection.py
```
