# Enterprise Master Data Management (MDM) Entity Resolution, Hybrid Blocking & Spec-Veto Golden-Record Agent Evaluator

Deterministic evaluator comparing four enterprise Master Data Management (`MDM`) entity resolution and deduplication architectures across 2.4M supplier and material records merged from 14 regional ERP instances:
- **Tier 1**: Legacy Levenshtein/Regex Rules + Manual Data Steward Review (`68.5%` F1, `1.4%` false-positive bad-merge rate, `95.0%` pair pruning, `$245,000/mo`)
- **Tier 2**: Pure Dense-Vector Cosine (`>=0.92`) Auto-Merge + Unconstrained LLM Prompt (`73.0%` F1, `13.6%` bad-merge corruption rate, `98.0%` pruning, `$390,000/mo`)
- **Tier 3**: Vector Top-10 Blocking + Holistic LLM Pairwise Judge (`84.8%` F1, `4.2%` bad-merge rate, `99.5%` pruning, `$128,000/mo`)
- **Tier 4**: Hybrid `LSH + Vector` Blocking (`99.94%` pruning) + LLM Spec-Attribute Extractor + Deterministic `Tax ID / DUNS / Spec-Inequality` Veto Gate + Survivorship Golden-Record Agent (`97.4%` F1, `0.0%` false-positive bad-merge rate, `$34,000/mo`)

## Usage
```bash
python experiments/enterprise_mdm_entity_resolution_supplier_product_dedup_agent_selection/run.py
python experiments/enterprise_mdm_entity_resolution_supplier_product_dedup_agent_selection/run.py --check
```
