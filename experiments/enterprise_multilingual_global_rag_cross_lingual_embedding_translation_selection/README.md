# Global Enterprise Multilingual & Cross-Lingual RAG Architecture Selection (`R509`)

Deterministic quantitative evaluation model comparing four global enterprise multilingual and cross-lingual RAG architectures across **15,000,000 documents and 12 operating languages**:
- **Tier 1**: English-Centric Monolingual Embedding + Naive BM25 Without Cross-Lingual Alignment
- **Tier 2**: Offline Pre-Translation of All 15M Documents into 12 Target Languages (`12x Index Explosion`)
- **Tier 3**: Synchronous Online LLM Query Translation (`Translate Query to English -> Search`) Without Glossary
- **Tier 4**: Single Native-Language Index (`BGE-M3 / Cohere Embed v3 / Multilingual-E5`) + `ICU/Jieba/Sudachi` Sparse BM25 + `2ms` Aho-Corasick Glossary Term Expansion + Cross-Lingual `bge-reranker-v2-m3`

## Verification

```bash
python experiments/enterprise_multilingual_global_rag_cross_lingual_embedding_translation_selection/run.py --check
python -m unittest tests/test_enterprise_multilingual_global_rag_cross_lingual_embedding_translation_selection.py
```
