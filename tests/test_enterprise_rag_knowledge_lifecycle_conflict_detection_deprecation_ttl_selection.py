import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseRagKnowledgeLifecycleSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_rag_knowledge_lifecycle_conflict_detection_deprecation_ttl_selection"
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
        t1 = tiers["tier1_append_only_vector_dump"]
        t4 = tiers["tier4_ingestion_nli_supersession_temporal_filter"]
        self.assertLess(t4["stale_or_contradictory_answer_rate_pct"], 0.5)
        self.assertGreater(t1["stale_or_contradictory_answer_rate_pct"], 20.0)
        self.assertGreater(t4["ingestion_conflict_detection_rate_pct"], 98.0)
        self.assertLess(
            t4["monthly_compliance_and_curation_tco_usd"],
            t1["monthly_compliance_and_curation_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-rag-knowledge-lifecycle-conflict-detection-deprecation-ttl"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
