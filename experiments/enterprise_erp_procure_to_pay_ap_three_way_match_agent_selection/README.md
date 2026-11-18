# Enterprise ERP Procure-to-Pay (P2P) Accounts Payable 3-Way Match, UoM Normalization & Anti-BEC Agent Evaluator

Deterministic evaluator comparing four enterprise Accounts Payable (`AP`) and Procure-to-Pay (`P2P`) 3-Way Match (`PO - Goods Receipt - Invoice`) architectures across 280,000 monthly vendor invoices, 14,000 suppliers, and 850,000 SAP S/4HANA / Oracle line items:
- **Tier 1**: Legacy Zonal OCR Templates + Manual AP Clerk 3-Way Matching in SAP `MIRO` (`26.5%` touchless STP, `0.35%` duplicate/overpayment rate, `$11.80 / invoice`, `$330,000/mo`)
- **Tier 2**: Prompt-Only VLM/LLM Reading Invoices & Directly Posting SAP Payment Vouchers (`78.0%` STP, `6.40%` overpayment / BEC bank-hijack rate, `$16.50 / invoice`, `$462,000/mo`)
- **Tier 3**: VLM Invoice Extraction + Vector Similarity Line-Item Matching (`61.0%` STP, `1.85%` overpayment / UoM error rate, `$4.90 / invoice`, `$137,000/mo`)
- **Tier 4**: Layout VLM + Semantic SKU/UoM Resolver + Deterministic 3-Way Match Tolerance Engine + Immutable SAP Vendor Bank-Master BEC Gate (`89.4%` STP, `0.0%` overpayment / BEC fraud rate, `$1.45 / invoice`, `$40,600/mo`)

## Usage
```bash
python experiments/enterprise_erp_procure_to_pay_ap_three_way_match_agent_selection/run.py
python experiments/enterprise_erp_procure_to_pay_ap_three_way_match_agent_selection/run.py --check
```
