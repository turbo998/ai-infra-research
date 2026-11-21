# Enterprise Supply Chain Control Tower, Time-Series Foundation Models & MILP Solver Agent Selection Guide: `Legacy APS` vs. `Chronos-Bolt / LightGBM + OR-Tools MILP Copilot`

> **Language / 语言**: [简体中文](./enterprise-supply-chain-control-tower-chronos-milp-agent-tco-selection-faq.md) | **English**  
> **Companion Interactive Study & Sizing Tools**: [📱 Interactive H5 Study Lab](../h5/supply-chain-control-tower-chronos-milp-agent-lab.html) · [🧮 HTML Sizer](../reports/enterprise-supply-chain-control-tower-chronos-milp-agent-tco-selection-sizer.html) · [🃏 7-Card Visual Deck](../reports/enterprise-supply-chain-control-tower-chronos-milp-agent-tco-selection-cards.html) · [📊 Reproducible Model](../../experiments/enterprise_supply_chain_control_tower_chronos_milp_agent_tco_selection/) · [🎓 Back to AI Infra H5 Hub](../h5/index.html)

![Enterprise Supply Chain Control Tower & MILP Solver Summary](../../assets/svg/enterprise-supply-chain-control-tower-chronos-milp-agent-tco-selection-faq-summary.svg)

---

## 1. Executive Summary (3-Minute Takeaway)

> **Bottom-Line Recommendation**: Never prompt a text LLM directly with historical sales CSVs to forecast demand or output purchase orders (`31.5%` WMAPE error and **`38.0%` violation rate** on MOQ, container capacity, and lead-time constraints).
> - **Recommended Production Architecture (`Tier 4 Neuro-Symbolic Control Tower: LLM Unstructured Disruption Sensor + Hybrid [LightGBM Mature SKUs + Chronos-Bolt Cold-Start SKUs] Quantile Forecasting + LLM Natural-Language-to-Pyomo/OR-Tools MILP Solver`)**:
>   1. **Reduces Forecast Error (`WMAPE`) from `26.4%` to `10.6%`** by combining `LightGBM` for promotion-heavy mature SKUs (`80%`) with zero-shot `Chronos-Bolt` for cold-start launches (`20%`) plus LLM-extracted supplier lead-time shocks;
>   2. **Guarantees `0.0%` Hard-Constraint Violations** via `OR-Tools / Gurobi` MILP optimization, cutting stockout rates from **`7.4%` to `1.8%`**, shrinking disruption replanning from **72 hours to `12 minutes`**, and saving **`$12.1M/year`** across a `$500M GMV` supply chain (`$6.4M/yr` vs. `$18.5M/yr`).

| Dimension | `Tier 1: Legacy ERP ARIMA + Excel` | `Tier 2: Pure Text LLM Forecasting` | `Tier 3: Commercial APS (Blue Yonder / SAP IBP)` | `Tier 4: LLM Sensor + Chronos/GBDT + MILP Solver (🏆)` |
|---|---|---|---|---|
| **Demand Forecast Error (`WMAPE`)** | `26.4%` ✕ | `31.5%` ✕ | `14.8%` ✓ | **`10.6%` (`LightGBM` + `Chronos-Bolt` + LLM shocks)** 🏆 |
| **Hard Constraint Violation Rate** | `4.5%` | **`38.0%` ✕ (Breaks MOQ/capacity)** | `0.0%` ✓ | **`0.0%` (`OR-Tools / Gurobi` exact MILP)** 🏆 |
| **Stockout Rate** | `7.4%` ✕ | `8.8%` ✕ | `3.1%` ✓ | **`1.8%`** 🏆 |
| **Disruption Replanning Latency** | `72.0 hrs` ✕ | `1.5 hrs` (Invalid plan) | `18.0 hrs` | **`0.2 hrs (12 min)`** 🏆 |
| **Annual Stockout + Carrying + Platform Cost** | `$18.5M / yr` | `$22.4M / yr` ✕ | `$11.2M / yr` | **`$6.4M / yr` (-65.4% / Saves $12.1M/yr)** 🏆 |

---

## 2. Provenance & Reproducibility

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)` — verified via [`experiments/enterprise_supply_chain_control_tower_chronos_milp_agent_tco_selection/run.py`](../../experiments/enterprise_supply_chain_control_tower_chronos_milp_agent_tco_selection/run.py).
