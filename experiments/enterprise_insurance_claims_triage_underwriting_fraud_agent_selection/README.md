# Enterprise Insurance Claims Triage, Multimodal Invoice Forensics, GNN Fraud & Deterministic Actuarial Adjudication Evaluator

Deterministic evaluator comparing four insurance claims adjudication and anti-fraud agent architectures across 420,000 monthly P&C/health claims, 1.8M invoice/damage images, and 1,200 active policy wordings & riders:
- **Tier 1**: Manual Claims Adjuster Review + Legacy Hardcoded Rule Engine (`22.0%` STP rate, `58.5%` fraud detection recall, `93.5%` payout accuracy, `$390,000/mo`)
- **Tier 2**: End-to-End Prompted VLM/LLM Directly Deciding Coverage & Payout Amount (`74.0%` STP rate, `46.0%` fraud recall, `68.4%` payout accuracy, `$480,000/mo`)
- **Tier 3**: VLM Invoice/Damage OCR + Standard Policy Wording Vector RAG (`58.5%` STP rate, `71.2%` fraud recall, `83.6%` payout accuracy, `$175,000/mo`)
- **Tier 4**: Pixel/EXIF Forgery Forensics + GNN Collusion Graph + Policy Rider AST + Deterministic Actuarial Payout Calculator + SIU/Adjuster HITL Agent (`82.5%` STP rate, `96.4%` fraud recall, `99.6%` payout accuracy, `$62,000/mo`)

## Usage
```bash
python experiments/enterprise_insurance_claims_triage_underwriting_fraud_agent_selection/run.py
python experiments/enterprise_insurance_claims_triage_underwriting_fraud_agent_selection/run.py --check
```
