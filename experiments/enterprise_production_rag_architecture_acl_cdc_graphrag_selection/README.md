# Enterprise Production RAG Architecture, Pre-Filter ACL & Selective Hybrid GraphRAG (`enterprise_production_rag_architecture_acl_cdc_graphrag_selection`)

Deterministic architectural evaluator for **SA Guide**: comparing **Naive Fixed Chunking + Post-Filter ACL**, **Full-Corpus Naive LLM GraphRAG**, **Layout Parent-Child + Hybrid BM25/Dense + Reranker + Pre-Filter Bitmap ACL**, and **Adaptive Hybrid RAG + Selective LazyGraphRAG Router**.

## Usage

```bash
python experiments/enterprise_production_rag_architecture_acl_cdc_graphrag_selection/run.py
python experiments/enterprise_production_rag_architecture_acl_cdc_graphrag_selection/run.py --check
```
