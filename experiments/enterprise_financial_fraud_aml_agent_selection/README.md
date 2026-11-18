# Enterprise Financial Fraud Detection, AML Investigation & Real-Time Transaction Risk Agent Evaluator 

Deterministic evaluator comparing four financial transaction fraud and Anti-Money Laundering (`AML / SAR`) agent architectures across 100,000,000 monthly transactions (3,000 TPS peak) and 120,000 monthly AML alerts:
- **Tier 1**: Legacy Static SQL Velocity Rules + Manual 8-System AML Investigation (`94.2%` false positive rate, `12.0 ms` P99 auth latency, `45.0 min` case review, `$185,000/mo`)
- **Tier 2**: Synchronous Inline LLM in Real-Time Payment Authorization Path (`38.0%` false positive rate, `1,450.0 ms` P99 auth latency [violates `<50ms` card network SLA!], `15.0 min` case review, `$420,000/mo`)
- **Tier 3**: Inline XGBoost (`14.0 ms`) + Unstructured Free-Text LLM Log Summary (`42.5%` false positive rate, `14.0 ms` P99 auth latency, `12.0 min` case review, `$68,000/mo`)
- **Tier 4**: Hot-Path `14ms` Feature-Store GNN/XGBoost + Warm-Path `0.4%` Grey-Zone `8B` Evaluator + Cold-Path Graph-RAG FinCEN SAR Copilot (`16.4%` false positive rate, `14.0 ms` P99 auth latency, `3.2 min` case review, `$24,500/mo`)

## Usage
```bash
python experiments/enterprise_financial_fraud_aml_agent_selection/run.py
python experiments/enterprise_financial_fraud_aml_agent_selection/run.py --check
```
