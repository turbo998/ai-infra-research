import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseSearchFederatedVsUnifiedIndexSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_search_cross_saas_federated_rag_vs_unified_index_connector_selection"
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
        t1 = tiers["tier1_pure_live_federated_api_fanout"]
        t4 = tiers["tier4_split_plane_85_unified_index_15_targeted_federated"]
        self.assertGreater(t4["cross_silo_ndcg_at_10_pct"], 96.0)
        self.assertLessEqual(t4["search_p99_latency_ms"], 400.0)
        self.assertEqual(t4["upstream_saas_429_throttle_rate_pct"], 0.0)
        self.assertLess(
            t4["monthly_platform_and_productivity_tco_usd"],
            t1["monthly_platform_and_productivity_tco_usd"] * 0.16,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-search-cross-saas-federated-rag-vs-unified-index-connector"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
