import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseMeetingIntelligenceSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_meeting_intelligence_diarization_action_item_crm_sync_selection"
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
        t1 = tiers["tier1_mixed_audio_generic_asr_blind_crm_write"]
        t4 = tiers["tier4_multimodal_diarization_hotword_boost_staged_crm"]
        self.assertLess(t4["speaker_attribution_error_der_pct"], 2.0)
        self.assertLess(t4["domain_jargon_sku_error_rate_pct"], 3.0)
        self.assertLess(t4["false_crm_jira_action_item_rate_pct"], 1.0)
        self.assertLess(t4["monthly_total_tco_usd"], t1["monthly_total_tco_usd"] * 0.15)

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-meeting-intelligence-diarization-action-item-crm-sync"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
