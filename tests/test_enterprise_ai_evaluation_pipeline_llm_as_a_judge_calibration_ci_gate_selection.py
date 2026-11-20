import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseAiEvaluationPipelineSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_ai_evaluation_pipeline_llm_as_a_judge_calibration_ci_gate_selection"
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
        t1 = tiers["tier1_uncalibrated_likert_1_to_10_full_suite_per_pr"]
        t4 = tiers["tier4_binary_rubric_swap_debiased_two_speed_ci_funnel"]
        self.assertGreater(t4["human_expert_agreement_rate_pct"], 96.0)
        self.assertLess(t4["position_and_verbosity_bias_error_pct"], 4.0)
        self.assertLessEqual(t4["pr_ci_gate_duration_minutes"], 2.0)
        self.assertLess(
            t4["monthly_eval_compute_and_regression_tco_usd"],
            t1["monthly_eval_compute_and_regression_tco_usd"] * 0.15,
        )

    def test_artifacts_exist(self) -> None:
        slug = "enterprise-ai-evaluation-pipeline-llm-as-a-judge-calibration-ci-gate"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
