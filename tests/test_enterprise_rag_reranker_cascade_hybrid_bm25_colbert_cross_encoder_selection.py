import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseRagRerankerCascadeSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_rag_reranker_cascade_hybrid_bm25_colbert_cross_encoder_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_script_check_and_invariants(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"], "synthetic + anonymized_reported")
        self.assertFalse(data["independently_reproduced"])
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_dense_vector_top5_only"]
        t4 = tiers["tier4_three_stage_rrf_colbert_cross_encoder_cutoff"]
        self.assertGreater(t4["ndcg_at_5_pct"], 96.0)
        self.assertLessEqual(t4["retrieval_and_rerank_p99_ms"], 45.0)
        self.assertGreaterEqual(t4["llm_context_token_reduction_pct"], 60.0)
        self.assertLess(
            t4["monthly_compute_and_llm_token_tco_usd"],
            t1["monthly_compute_and_llm_token_tco_usd"] * 0.2,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-rag-reranker-cascade-hybrid-bm25-colbert-cross-encoder"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
