import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseRAGVectorDBMilvusQdrantPgvectorAISearchSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_rag_vector_db_milvus_qdrant_pgvector_ai_search_hnsw_diskann_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "enterprise_rag_vector_db_milvus_qdrant_pgvector_ai_search_hnsw_diskann_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Milvus_AI_Search_Tiered_DiskANN_Hybrid_RRF"]["Required_Cluster_DRAM_GB"],
            tiers["Tier_1_In_Memory_HNSW_FP32_Uncompressed"]["Required_Cluster_DRAM_GB"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "enterprise-rag-vector-db-milvus-qdrant-pgvector-ai-search-hnsw-diskann-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "enterprise-rag-vector-db-milvus-qdrant-pgvector-ai-search-hnsw-diskann-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "enterprise-rag-vector-db-milvus-qdrant-pgvector-ai-search-hnsw-diskann-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "enterprise-rag-vector-db-milvus-qdrant-pgvector-ai-search-hnsw-diskann-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "enterprise-rag-vector-db-milvus-qdrant-pgvector-ai-search-hnsw-diskann-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
