import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseLlmEvaluationMetricsSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_llm_evaluation_metrics_ttft_tpot_e2el_goodput_sla_selection"
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
        t1 = tiers["tier1_max_batch_raw_tps_tuning_ignoring_p99_ttft_and_itl_sla"]
        t4 = tiers["tier4_dual_sla_goodput_pareto_knee_controller_pd_and_admission_gate"]
        self.assertGreater(
            t4["sla_compliant_goodput_tps_per_gpu"],
            t1["sla_compliant_goodput_tps_per_gpu"] * 5.0,
        )
        self.assertLess(t4["p99_itl_ms"], 20.0)
        self.assertLess(
            t4["effective_cost_per_1m_goodput_tokens_usd"],
            t1["effective_cost_per_1m_goodput_tokens_usd"] * 0.25,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-llm-evaluation-metrics-ttft-tpot-e2el-goodput-sla"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
