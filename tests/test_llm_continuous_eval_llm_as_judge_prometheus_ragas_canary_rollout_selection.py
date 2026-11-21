import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMContinuousEvalLLMAsJudgePrometheusRAGASSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_continuous_eval_llm_as_judge_prometheus_ragas_canary_rollout_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_continuous_eval_llm_as_judge_prometheus_ragas_canary_rollout_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Tiered_DeBERTa_NLI_Plus_Prometheus14B_Spot_Cascade"]["Monthly_Eval_Infrastructure_TCO_USD"],
            tiers["Tier_1_100Pct_Frontier_Commercial_API_Judge"]["Monthly_Eval_Infrastructure_TCO_USD"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-continuous-eval-llm-as-judge-prometheus-ragas-canary-rollout-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-continuous-eval-llm-as-judge-prometheus-ragas-canary-rollout-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-continuous-eval-llm-as-judge-prometheus-ragas-canary-rollout-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-continuous-eval-llm-as-judge-prometheus-ragas-canary-rollout-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-continuous-eval-llm-as-judge-prometheus-ragas-canary-rollout-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
