# Enterprise Supply Chain Control Tower, Demand Forecasting & Disruption Mitigation Agent Evaluator 

Deterministic evaluator comparing four supply chain demand forecasting and inventory replenishment architectures across 250,000 SKUs, 40 warehouses, and 15,000 monthly purchase orders:
- **Tier 1**: Legacy ERP Static Moving-Average + Manual Excel Expediting (`68.5%` forecast accuracy `1-WAPE`, `4.2%` MOQ violation rate, `72.0 h` disruption recovery, `$145,000/mo`)
- **Tier 2**: Pure Generative LLM Prompted to Predict Time-Series Numbers & Order Quantities (`62.0%` forecast accuracy, `18.5%` MOQ/capacity violation rate, `1.5 h` recovery, `$210,000/mo`)
- **Tier 3**: Time-Series Model (`Chronos / LightGBM`) + Unconstrained LLM Order Recommendation (`85.4%` forecast accuracy, `6.8%` MOQ violation rate, `2.0 h` recovery, `$64,000/mo`)
- **Tier 4**: LLM Exogenous Disruption Covariates + `Chronos / LightGBM` Forecaster + Deterministic `OR-Tools / Gurobi MILP` Solver + ERP Purchase-Order HITL Agent (`91.2%` forecast accuracy, `0.0%` MOQ violation rate, `0.25 h [15 min]` recovery, `$22,000/mo`)

## Usage
```bash
python experiments/enterprise_supply_chain_control_tower_agent_selection/run.py
python experiments/enterprise_supply_chain_control_tower_agent_selection/run.py --check
```
