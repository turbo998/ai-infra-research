import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmFirstPrinciplesRooflineGoodputSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_first_principles_roofline_6nd_kv_xprof_goodput_triage_selection"
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
        t1 = tiers["tier1_blind_trial_and_error_scaling_no_roofline_or_xprof"]
        t4 = tiers["tier4_roofline_sized_xprof_overlapped_3factor_goodput_pd_disaggregated"]
        self.assertGreater(t4["sustained_training_mfu_pct"], 58.0)
        self.assertGreater(t4["ml_productivity_goodput_pct"], 95.0)
        self.assertGreater(t4["decode_mbu_bandwidth_utilization_pct"], 75.0)
        self.assertLess(
            t4["monthly_wasted_cluster_and_serving_tco_usd"],
            t1["monthly_wasted_cluster_and_serving_tco_usd"] * 0.25,
        )

    def test_artifacts_exist(self) -> None:
        slug = "llm-first-principles-roofline-6nd-kv-xprof-goodput-triage"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
