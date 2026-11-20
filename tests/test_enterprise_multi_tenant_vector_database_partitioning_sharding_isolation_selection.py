import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseMultiTenantVectorDbSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_multi_tenant_vector_database_partitioning_sharding_isolation_selection"
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
        t1 = tiers["tier1_monolithic_flat_hnsw_scalar_filter"]
        t2 = tiers["tier2_collection_per_tenant_5000_indexes"]
        t4 = tiers["tier4_tiered_partition_key_subgraph_plus_vip_shards"]
        self.assertGreater(t4["small_tenant_recall_at_10_pct"], 98.0)
        self.assertLess(t1["small_tenant_recall_at_10_pct"], 60.0)
        self.assertLessEqual(t4["small_tenant_p99_latency_ms"], 20.0)
        self.assertLess(t4["monthly_vector_cluster_tco_usd"], t2["monthly_vector_cluster_tco_usd"] * 0.1)

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-multi-tenant-vector-database-partitioning-sharding-isolation"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
