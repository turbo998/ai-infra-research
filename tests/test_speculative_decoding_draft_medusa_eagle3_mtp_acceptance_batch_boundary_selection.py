import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestSpeculativeDecodingDraftMedusaEagle3MtpSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "speculative_decoding_draft_medusa_eagle3_mtp_acceptance_batch_boundary_selection"
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
        t1 = tiers["tier1_standard_autoregressive_1_token_per_step_baseline"]
        t4 = tiers["tier4_native_mtp_or_eagle3_dynamic_tree_plus_adaptive_batch_cutoff"]
        self.assertGreater(t4["mean_accepted_tokens_per_step_tau"], 3.0)
        self.assertLess(t4["achieved_p99_effective_tpot_ms"], 10.0)
        self.assertLess(
            t4["cost_per_1m_tokens_usd"],
            t1["cost_per_1m_tokens_usd"] * 0.50,
        )

    def test_artifacts_exist(self) -> None:
        slug = "speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
