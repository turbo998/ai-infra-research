# Enterprise AI Caching Architecture Selection: Semantic Cache vs. Provider Prompt Caching vs. Prefix `KV Cache` (`R477`)

Deterministic evaluation suite comparing four enterprise AI caching architectures across `60,000,000` monthly LLM requests (`10,500` average prompt prefix tokens, `35%` single-turn FAQ queries, `65%` multi-turn Agent & RAG workflows):

1. **Tier 1**: Exact-Match SHA-256 String Cache + Dynamic Timestamp at Top of System Prompt (`4.2%` token savings, `1,420ms` P95 TTFT, `$182,000/mo` TCO)
2. **Tier 2**: Naive Global Vector Semantic Cache (`Cosine > 0.88` Without Entity/ACL Guard) (`48.0%` token savings, `14.8%` semantic collision wrong-answer rate, `9.4%` cross-tenant ACL leak rate, `$145,000/mo` TCO incl. rework)
3. **Tier 3**: Cloud Provider Prompt Caching Only (`Azure OpenAI / Anthropic / Gemini` Static Prefix Ordering) (`56.5%` token savings, `0.0%` collision rate, `490ms` P95 TTFT, `$84,000/mo` TCO)
4. **Tier 4**: Two-Tier ACL-Scoped Entity-Guarded Semantic Cache (`Redis + Cross-Encoder`) + Canonical Prefix-Ordered `KV Cache` (`Provider / SGLang Radix / vLLM APC`) (`78.4%` input token cost reduction, `0.0%` semantic collision rate, `0.0%` ACL leak rate, `195ms` P95 TTFT, `$41,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_ai_caching_prefix_kv_cache_semantic_cache_redis_selection/run.py
python experiments/enterprise_ai_caching_prefix_kv_cache_semantic_cache_redis_selection/run.py --check
python -m unittest tests/test_enterprise_ai_caching_prefix_kv_cache_semantic_cache_redis_selection.py
```
