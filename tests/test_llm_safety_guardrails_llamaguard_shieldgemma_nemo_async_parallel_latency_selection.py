import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMSafetyGuardrailsAsyncParallelLatencySelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_safety_guardrails_llamaguard_shieldgemma_nemo_async_parallel_latency_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_safety_guardrails_llamaguard_shieldgemma_nemo_async_parallel_latency_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_FastPath_Plus_Speculative_Parallel_Streaming_Guard"]["Added_Guardrail_TTFT_Penalty_ms"],
            tiers["Tier_1_Sync_Serial_Pre_And_Full_Post_LLM_Guard"]["Added_Guardrail_TTFT_Penalty_ms"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-safety-guardrails-llamaguard-shieldgemma-nemo-async-parallel-latency-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-safety-guardrails-llamaguard-shieldgemma-nemo-async-parallel-latency-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-safety-guardrails-llamaguard-shieldgemma-nemo-async-parallel-latency-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-safety-guardrails-llamaguard-shieldgemma-nemo-async-parallel-latency-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-safety-guardrails-llamaguard-shieldgemma-nemo-async-parallel-latency-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
