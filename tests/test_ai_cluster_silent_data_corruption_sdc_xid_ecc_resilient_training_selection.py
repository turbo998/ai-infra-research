import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAiClusterSilentDataCorruptionSdcXidResilientSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "ai_cluster_silent_data_corruption_sdc_xid_ecc_resilient_training_selection"
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
        t1 = tiers["tier1_reactive_manual_restart_without_sdc_or_xid_auto_cordon"]
        t4 = tiers["tier4_inline_sdc_golden_hash_pre_allreduce_norm_gate_and_42s_p2p_failover"]
        self.assertGreater(t4["effective_cluster_goodput_pct"], 97.5)
        self.assertEqual(t4["un_caught_sdc_checkpoint_poison_incidents_per_mo"], 0)
        self.assertLess(
            t4["monthly_2048gpu_wasted_compute_tco_usd"],
            t1["monthly_2048gpu_wasted_compute_tco_usd"] * 0.10,
        )

    def test_artifacts_exist(self) -> None:
        slug = "ai-cluster-silent-data-corruption-sdc-xid-ecc-resilient-training"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
