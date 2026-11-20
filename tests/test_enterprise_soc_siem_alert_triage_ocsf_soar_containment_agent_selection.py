import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseSocSiemAlertTriageSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_soc_siem_alert_triage_ocsf_soar_containment_agent_selection"
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
        t1 = tiers["tier1_per_alert_raw_llm_unfettered_soar_isolation"]
        t4 = tiers["tier4_ocsf_entity_clustering_spotlighting_opa_soar_guardrail"]
        self.assertLess(t4["mean_time_to_respond_mttr_minutes"], 4.0)
        self.assertEqual(t4["log_prompt_injection_bypass_rate_pct"], 0.0)
        self.assertEqual(t4["monthly_self_inflicted_prod_outages"], 0.0)
        self.assertLess(
            t4["monthly_soc_staffing_token_and_outage_tco_usd"],
            t1["monthly_soc_staffing_token_and_outage_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-soc-siem-alert-triage-ocsf-soar-containment-agent"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
