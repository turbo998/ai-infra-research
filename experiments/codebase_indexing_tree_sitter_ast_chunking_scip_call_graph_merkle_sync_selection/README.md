# Enterprise Codebase Indexing & Repo Retrieval Selection Experiment (`Tree-sitter AST` + `SCIP Call-Graph` + `Merkle Sync`)

Reproducible analytical model comparing 4 codebase indexing architectures across a `50,000-file` (`15M LOC`) enterprise monorepo with `1,000` developers:
- **Tier 1 (`Naive Fixed-Token Line Split + Full Re-Index`)**: `48.5%` SWE localization `Recall@10`, `2,520 sec` (`42 min`) branch-switch re-index, `$48,000/month` GPU cost.
- **Tier 2 (`Tree-sitter AST Chunking + Dense Vector Only`)**: `74.2%` `Recall@10`, `45.0 sec` sync.
- **Tier 3 (`Tree-sitter cAST with Parent Header + Hybrid Trigram/BM25 + Merkle-Tree Sync`)**: `84.6%` `Recall@10`, `0.18 sec` (`180 ms`) branch sync (`99.8%` content-addressable chunk reuse).
- **Tier 4 (`Tree-sitter cAST + Hybrid Trigram/Vector + SCIP/LSP 1-Hop Call-Graph + Merkle Sync`, Recommended)**: **`94.8%` `Recall@10`**, `0.18 sec` branch sync, and `$1,680/month` (`-96.5%` indexing GPU cost vs. Tier 1).

## Run & Verify

```bash
python experiments/codebase_indexing_tree_sitter_ast_chunking_scip_call_graph_merkle_sync_selection/run.py
python experiments/codebase_indexing_tree_sitter_ast_chunking_scip_call_graph_merkle_sync_selection/run.py --check
```
