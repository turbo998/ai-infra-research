# Enterprise Global Trade & Customs Compliance: 10-Digit HS Code GRI Tree, FTA RVC & OFAC/BIS Export Control Evaluator

Deterministic evaluator comparing four global trade compliance, 10-digit Harmonized System (`HS / HTS`) tariff classification, and export-control (`OFAC / BIS ECCN`) screening architectures across 95,000 monthly customs declarations, 320,000 SKUs, and 55,000 counterparties:
- **Tier 1**: Manual Customs Broker Tariff Lookup + Spreadsheet FTA Origin Calculation (`89.0%` 10-digit HS accuracy, `1.6%` sanctions/ECCN miss rate, `65.0 min` prep time, `$275,000/mo`)
- **Tier 2**: Prompt-Only LLM Generating 10-Digit HS Codes & Judging Export Controls Directly (`62.5%` HS accuracy, `12.4%` sanctions/ECCN miss rate, `2.2 min` prep time, `$410,000/mo`)
- **Tier 3**: Flat Vector Search over HS Descriptions + Exact Keyword Sanction Screening (`76.8%` HS accuracy, `4.5%` sanctions miss rate, `14.0 min` prep time, `$132,000/mo`)
- **Tier 4**: Hierarchical `WCO GRI 1–6` & Chapter-Note Tree + Deterministic BOM `RVC` Calculator + `OFAC 50%-Rule` Ownership Graph Agent (`96.6%` HS accuracy, `0.0%` sanctions/ECCN miss rate, `1.8 min` prep time, `$39,000/mo`)

## Usage
```bash
python experiments/enterprise_customs_tariff_hs_code_export_control_agent_selection/run.py
python experiments/enterprise_customs_tariff_hs_code_export_control_agent_selection/run.py --check
```
