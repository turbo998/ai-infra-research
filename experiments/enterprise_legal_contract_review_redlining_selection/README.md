# Enterprise Legal Contract Review, Redlining & Clause Playbook Compliance Agent Architecture Evaluator (Round 403)

Deterministic evaluator comparing four corporate legal contract review and DOCX redlining architectures across 25,000 commercial contracts/month (45 pages / 28,000 tokens avg):
- **Tier 1**: Naive `512-Token` Chunked RAG + Plain-Text Chat Copy-Paste (`58.4%` risk recall, `14.5%` edit error, `0.0%` DOCX Track-Changes fidelity, `$18,500/mo`)
- **Tier 2**: `128K` Monolithic Single-Prompt Dump + Full-Paragraph Rewrite (`82.6%` risk recall, `8.2%` edit error, `35.0%` DOCX fidelity, `$46,000/mo`)
- **Tier 3**: Clause-Segmented Extraction + Sequential 60-Rule LLM Scan (`91.2%` risk recall, `3.4%` edit error, `78.0%` DOCX fidelity, `$29,400/mo`)
- **Tier 4**: Hierarchical Legal AST + Definitions/Cross-Reference Graph Pinning + 3-Tier Playbook Fallback Matrix + Minimal-Span OpenXML Track-Changes (`w:ins` / `w:del`) (`98.6%` risk recall, `0.3%` edit error, `99.8%` DOCX fidelity, `$6,800/mo`)

## Usage
```bash
python experiments/enterprise_legal_contract_review_redlining_selection/run.py
python experiments/enterprise_legal_contract_review_redlining_selection/run.py --check
```
