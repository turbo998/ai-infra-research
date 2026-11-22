# Enterprise Long-Prompt Compression & KV Eviction (`LLMLingua-2` vs. `StreamingLLM / SnapKV` vs. `Prefix-Preserving Radix Cache`) (`#484`)

Reproducible analytical model demonstrating why query-aware prompt compression (`LLMLingua`) over shared prefixes destroys `RadixAttention / Prefix Cache` block hashes (`620ms TTFT` vs. `192ms`), and establishing the Prefix-Preserving hybrid standard.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_prompt_compression_kv_eviction_llmlingua_snapkv_radix_selection/run.py --check`
