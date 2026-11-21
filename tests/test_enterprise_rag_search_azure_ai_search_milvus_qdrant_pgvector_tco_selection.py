from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseRagSearchAzureAiSearchMilvusQdrantPgvectorTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_rag_search_azure_ai_search_milvus_qdrant_pgvector_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["cluster_ram_gb"], 100)
        self.assertGreater(data["tiers"][3]["recall_at_5_pct"], 94.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-rag-search-azure-ai-search-milvus-qdrant-pgvector-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-rag-search-azure-ai-search-milvus-qdrant-pgvector-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-rag-search-azure-ai-search-milvus-qdrant-pgvector-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "rag-azure-ai-search-milvus-qdrant-pgvector-sizer-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-rag-search-azure-ai-search-milvus-qdrant-pgvector-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-rag-search-azure-ai-search-milvus-qdrant-pgvector-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
