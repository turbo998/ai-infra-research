from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestCodebaseIndexingTreeSitterAstChunkingScipCallGraphMerkleSyncSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "codebase_indexing_tree_sitter_ast_chunking_scip_call_graph_merkle_sync_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["swe_localization_recall_at_10_pct"], 94.0)
        self.assertLess(data["tiers"][3]["branch_switch_sync_latency_sec"], 0.5)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "codebase-indexing-tree-sitter-ast-chunking-scip-call-graph-merkle-sync-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "codebase-indexing-tree-sitter-ast-chunking-scip-call-graph-merkle-sync-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "codebase-indexing-tree-sitter-ast-chunking-scip-call-graph-merkle-sync-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "codebase-indexing-tree-sitter-ast-chunking-scip-call-graph-merkle-sync-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "codebase-indexing-tree-sitter-ast-chunking-scip-call-graph-merkle-sync-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
