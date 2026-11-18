# Enterprise ESG Carbon Accounting (Scope 1-2-3), CSRD/CBAM Compliance, Ecoinvent Resolver & ISAE 3000 Audit Agent Evaluator

Deterministic evaluator comparing four enterprise ESG carbon accounting (`GHG Protocol Scope 1-2-3`), EU `CSRD / CBAM`, and `ISAE 3000` external assurance architectures across 85 global sites, 4,200 suppliers, and 380,000 utility/freight/BOM records:
- **Tier 1**: Manual Excel Collection + Crude Spend-Based EEIO Financial Averages (`78.0%` assurance accuracy, `4.2%` unit/factor error rate, `64.0%` audit traceability, `$235,000/mo`)
- **Tier 2**: Prompt-Only LLM Using Memorized Emission Factors & Mental Math (`58.5%` accuracy, `26.8%` emission-factor / unit hallucination rate, `22.0%` traceability, `$380,000/mo`)
- **Tier 3**: ESG Document Vector RAG + Unconstrained LLM Table Calculation (`82.5%` accuracy, `8.6%` error rate, `76.5%` traceability, `$115,000/mo`)
- **Tier 4**: Multimodal Utility/BOM Extractor + Physical Unit Normalizer + Constrained `Ecoinvent v3.10 / DEFRA` Resolver + Deterministic GHG Engine & `ISAE 3000` Lineage Agent (`99.3%` accuracy, `0.0%` hallucination rate, `100.0%` audit traceability, `$32,000/mo`)

## Usage
```bash
python experiments/enterprise_esg_carbon_accounting_scope3_csrd_audit_agent_selection/run.py
python experiments/enterprise_esg_carbon_accounting_scope3_csrd_audit_agent_selection/run.py --check
```
