import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMServingAdmissionControlBackpressureSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_serving_admission_control_backpressure_queue_shedding_ttft_sla_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_serving_admission_control_backpressure_queue_shedding_ttft_sla_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_Predictive_TTFT_Admission_Plus_Graceful_SLM_Downgrade"]["Effective_Goodput_Meeting_SLA_RPS"],
            tiers["Tier_1_Unbounded_FIFO_Queue_Plus_Client_Retry_Storm"]["Effective_Goodput_Meeting_SLA_RPS"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
