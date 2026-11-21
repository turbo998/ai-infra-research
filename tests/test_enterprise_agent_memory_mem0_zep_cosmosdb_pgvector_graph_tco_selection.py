from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAgentMemoryMem0ZepCosmosdbPgvectorGraphTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_agent_memory_mem0_zep_cosmosdb_pgvector_graph_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["temporal_state_update_accuracy_pct"], 94.0)
        self.assertLessEqual(data["tiers"][3]["memory_tokens_per_turn"], 1000)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-agent-memory-mem0-zep-cosmosdb-pgvector-graph-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-agent-memory-mem0-zep-cosmosdb-pgvector-graph-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-agent-memory-mem0-zep-cosmosdb-pgvector-graph-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "agent-memory-mem0-zep-cosmosdb-pgvector-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-agent-memory-mem0-zep-cosmosdb-pgvector-graph-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-agent-memory-mem0-zep-cosmosdb-pgvector-graph-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
