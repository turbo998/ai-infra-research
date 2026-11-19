# Enterprise Indirect Tax (`VAT / GST / Sales & Use Tax`), `STC` Classifier & `Peppol e-Invoicing` Agent (`R464`)

Deterministic evaluation suite comparing four enterprise indirect tax (`VAT / GST / US Sales & Use Tax`), Accounts Payable (`AP`) Consumer Use-Tax, exemption certificate management, and real-time `e-Invoicing` (`Peppol / ViDA`) architectures across `14M` annual transactions (`13,000` tax jurisdictions, `35,000` monthly AP invoices, `28,000` exemption certificates):

1. **Tier 1**: Static ERP Tax Rate Tables + Manual Exemption Binders + Monthly Batch Filing (`8.4%` audit assessment error, `12.6%` AP use-tax error, `14.2%` e-invoice rejection, `$780,000/mo` cost)
2. **Tier 2**: Prompt-Only General LLM Tax Rate & Statute Chatbot (`16.8%` tax assessment error, `18.2%` AP use-tax error, `22.0%` e-invoice rejection, `$920,000/mo` cost)
3. **Tier 3**: Tax Engine API + Manual Catch-All SKU Mapping (`3.1%` assessment error, `6.4%` AP use-tax error, `4.5%` e-invoice rejection, `$310,000/mo` cost)
4. **Tier 4**: Constrained `STC` Taxonomy Classifier + Rooftop-Geo Tax Engine + `AP Use-Tax AST` + `Peppol UBL` Gate Agent (`0.0%` assessment error, `0.2%` AP use-tax variance, `0.0%` e-invoice rejection, `0.0%` invalid certificate escape, `$64,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_tax_vat_sales_use_nexus_einvoicing_compliance_agent_selection/run.py
python experiments/enterprise_tax_vat_sales_use_nexus_einvoicing_compliance_agent_selection/run.py --check
python -m unittest tests/test_enterprise_tax_vat_sales_use_nexus_einvoicing_compliance_agent_selection.py
```
