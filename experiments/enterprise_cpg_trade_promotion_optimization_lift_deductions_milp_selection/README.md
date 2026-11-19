# Enterprise `CPG` Trade Promotion Optimization (`TPO`), True Lift Decomposition & Deductions Agent (`R463`)

Deterministic evaluation suite comparing four Consumer Packaged Goods (`CPG`) Revenue Growth Management (`RGM`), Trade Promotion Optimization (`TPO`), and retailer chargeback deduction-audit architectures across a `$180M/yr` trade-spend budget (`420` SKUs, `18` retail banners, `42,000` annual deduction claims):

1. **Tier 1**: Last-Year Roll-Forward Excel Calendar + Unaudited Retailer Deductions (`58.4%` negative-ROI promo share, `11.6%` invalid deduction leakage, `14.2%` promo OOS, `$1,380,000/mo` waste)
2. **Tier 2**: Prompt-Only General LLM Promo Calendar Planner & Dispute Writer (`54.0%` negative-ROI promo share, `13.2%` deduction leakage, `33.0%` net margin MAPE, `$1,490,000/mo` waste)
3. **Tier 3**: Own-Price Elasticity Regression Only (`24.5%` negative-ROI promo share, `5.4%` deduction leakage, `11.8%` net margin MAPE, `$510,000/mo` waste)
4. **Tier 4**: Hierarchical Cross-Elasticity + Pantry-Dip Decomposition + `JBP Calendar MILP` + `EDI 812` Deduction Agent (`3.2%` negative-ROI promo share, `0.2%` deduction leakage, `0.9%` promo OOS, `2.4%` net margin MAPE, `$115,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_cpg_trade_promotion_optimization_lift_deductions_milp_selection/run.py
python experiments/enterprise_cpg_trade_promotion_optimization_lift_deductions_milp_selection/run.py --check
python -m unittest tests/test_enterprise_cpg_trade_promotion_optimization_lift_deductions_milp_selection.py
```
