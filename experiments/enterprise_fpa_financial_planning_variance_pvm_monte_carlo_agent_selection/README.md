# Enterprise FP&A (`PVM-FX` Variance Decomposition, 3-Statement Driver DAG & Monte Carlo Cash Agent) (`R453`)

Deterministic evaluation suite comparing four enterprise Financial Planning & Analysis (`FP&A`), multi-ERP consolidation, and rolling cash-forecasting architectures across a `$1.2B/yr` enterprise (`28` global legal entities, `18,000` SKUs/cost centers, `14` currencies):

1. **Tier 1**: Manual Multi-ERP Excel Consolidation & Static Annual Budget (`14.5 days` close cycle, `4.2%` reconciliation error, `16.8%` cash MAPE, `22.5%` unexplained PVM residual, `$620,000/mo` cost)
2. **Tier 2**: Prompt-Only General LLM Raw GL/ERP Table Summarizer (`11.2 days` cycle, `19.4%` reconciliation error, `21.0%` cash MAPE, `34.0%` unexplained PVM residual, `$740,000/mo` cost)
3. **Tier 3**: BI Dashboard RAG + Unconstrained Time-Series Extrapolation (`5.4 days` cycle, `1.8%` reconciliation error, `8.9%` cash MAPE, `9.4%` unexplained PVM residual, `$295,000/mo` cost)
4. **Tier 4**: Exact `PVM-FX` Shapley Bridge + 3-Statement Driver DAG + Correlated Cholesky Monte Carlo Agent (`0.4 days` cycle, `0.0%` reconciliation error, `2.3%` cash MAPE, `0.0%` unexplained PVM residual, `$68,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_fpa_financial_planning_variance_pvm_monte_carlo_agent_selection/run.py
python experiments/enterprise_fpa_financial_planning_variance_pvm_monte_carlo_agent_selection/run.py --check
python -m unittest tests/test_enterprise_fpa_financial_planning_variance_pvm_monte_carlo_agent_selection.py
```
