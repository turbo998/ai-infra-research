import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestMoeArchitectureDeepEpIbgdaDualPipeEplbSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "moe_architecture_deepep_ibgda_dualpipe_eplb_load_balancing_selection"
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
        t1 = tiers["tier1_standard_nccl_alltoall_without_overlap_or_eplb"]
        t4 = tiers["tier4_deepep_ibgda_dual_batch_overlap_plus_hierarchical_eplb"]
        self.assertLess(t4["exposed_alltoall_communication_stall_pct"], 2.5)
        self.assertLess(t4["hot_expert_straggler_skew_ratio"], 1.15)
        self.assertGreater(
            t4["sla_compliant_goodput_tps_per_gpu"],
            t1["sla_compliant_goodput_tps_per_gpu"] * 4.0,
        )

    def test_artifacts_exist(self) -> None:
        slug = "moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
