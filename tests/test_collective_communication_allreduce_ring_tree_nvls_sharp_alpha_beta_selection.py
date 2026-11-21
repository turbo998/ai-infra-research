import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestCollectiveCommunicationRingTreeNvlsSharpSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "collective_communication_allreduce_ring_tree_nvls_sharp_alpha_beta_selection"
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
        t1 = tiers["tier1_forced_flat_ring_allreduce_across_all_sizes_and_1024_gpus"]
        t4 = tiers["tier4_hierarchical_nvls_plus_inter_node_sharpv3_collnet_and_flight_recorder"]
        self.assertLess(t4["small_4mb_allreduce_latency_us"], 60.0)
        self.assertGreater(
            t4["large_4gb_allreduce_algbw_gb_s"],
            t1["large_4gb_allreduce_algbw_gb_s"] * 1.70,
        )

    def test_artifacts_exist(self) -> None:
        slug = "collective-communication-allreduce-ring-tree-nvls-sharp-alpha-beta"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
