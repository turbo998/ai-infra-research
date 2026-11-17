# Enterprise E-Commerce Search, Personalized Recommendation & Shopping Agent Architecture Evaluator (Round 402)

Deterministic evaluator comparing four retail & e-commerce search/recommendation architectures across 5,000,000 SKUs and 60,000,000 monthly queries (1,000 RPS peak):
- **Tier 1**: Legacy BM25 Keyword Search + Static Rule Boost (`2.15%` long-tail CVR, `0.0%` price/stock hallucination, `35.0 ms` P95 latency, `$8,400/mo`)
- **Tier 2**: Synchronous `70B` LLM Conversational Search Blocking Every Query (`2.60%` long-tail CVR, `6.8%` price/stock hallucination, `2,850.0 ms` P95 latency, `$148,000/mo`)
- **Tier 3**: Synchronous `8B` LLM Query Rewriting + Hybrid BM25/Vector Search (`3.45%` long-tail CVR, `0.8%` price/stock hallucination, `380.0 ms` P95 latency, `$32,000/mo`)
- **Tier 4**: Offline LLM Catalog Enrichment + Sub-15ms `0.5B` Intent Parser + Two-Tower Personalized Recall + Real-Time Redis Inventory/Price Pinning + Async Streaming Shopping Advisor (`4.68%` long-tail CVR, `0.0%` price/stock hallucination, `62.0 ms` P95 latency, `$12,600/mo`)

## Usage
```bash
python experiments/enterprise_ecommerce_search_recommendation_agent_selection/run.py
python experiments/enterprise_ecommerce_search_recommendation_agent_selection/run.py --check
```
