# Enterprise Codebase Indexing & `SWE-Agent` (`Tree-Sitter AST + SCIP Call Graph + Merkle Sync`) Sizer (`#529`)

Reproducible analytical model comparing **Fixed 512-Token Text Chunking (`46.5%` cross-file bug localization recall)**, **128K Full-Directory Context Stuffing (`115K` tokens/query, `48 GPUs` ✕)**, and **`Tree-Sitter AST` Function/Class Chunking + `SCIP / LSIF` 2-Hop Caller/Callee Graph Expansion + `Merkle Tree` Incremental Git Sync + Sandboxed Unit-Test Gate (`94.6%` recall, `89.4%` PR test pass rate, `6,400` tokens/query, `8x H200 GPUs`, `-83.3%` GPUs 🏆)**.

## Run

```bash
python experiments/enterprise_codebase_rag_treesitter_ast_scip_callgraph_swe_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-codebase-rag-treesitter-ast-scip-callgraph-swe-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-codebase-rag-treesitter-ast-scip-callgraph-swe-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/codebase-rag-treesitter-ast-scip-callgraph-swe-lab.html](../../docs/h5/codebase-rag-treesitter-ast-scip-callgraph-swe-lab.html)
