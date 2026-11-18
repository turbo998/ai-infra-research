# Enterprise Wealth Management, Equity Research (SEC 10-K / Earnings Call) & Compliance Advisory Agent Evaluator

Deterministic evaluator comparing four institutional equity research and wealth management advisory agent architectures across 3,500 global equities, 28,000 SEC 10-K/10-Q filings & earnings call transcripts, and 18,000 monthly research queries:
- **Tier 1**: Manual Equity Analyst `Ctrl+F` over PDF Filings + Manual Excel Copy-Paste (`94.2%` metric accuracy, `3.8%` unit/copy error rate, `240.0 min` turnaround, `$165,000/mo`)
- **Tier 2**: Naive 512-Token Vector RAG + LLM Mental Math (`63.5%` metric accuracy, `31.4%` numerical/unit hallucination rate, `1.8 min` turnaround, `$195,000/mo`)
- **Tier 3**: 128K Long-Context Filing Stuffing without XBRL Fact Store or Python Sandbox (`81.6%` metric accuracy, `11.2%` numerical error rate, `3.2 min` turnaround, `$88,000/mo`)
- **Tier 4**: Deterministic XBRL/SQL Fact Store + Hierarchical 10-K/Transcript RAG + Python Sandbox Math + SEC/FINRA Restricted-List Gate (`99.4%` metric accuracy, `0.0%` numerical hallucination rate, `1.4 min` turnaround, `$24,000/mo`)

## Usage
```bash
python experiments/enterprise_wealth_management_investment_research_compliance_agent_selection/run.py
python experiments/enterprise_wealth_management_investment_research_compliance_agent_selection/run.py --check
```
