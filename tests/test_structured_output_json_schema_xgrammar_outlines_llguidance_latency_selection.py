import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestStructuredOutputXGrammarOutlinesLLGuidanceSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "structured_output_json_schema_xgrammar_outlines_llguidance_latency_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "structured_output_json_schema_xgrammar_outlines_llguidance_latency_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_XGrammar_LLGuidance_PDA_Async_Overlap"]["Per_Token_Mask_CPU_Overhead_ms"],
            tiers["Tier_2_Legacy_Per_Step_Regex_FSM_Unoptimized"]["Per_Token_Mask_CPU_Overhead_ms"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "structured-output-json-schema-xgrammar-outlines-llguidance-latency-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "structured-output-json-schema-xgrammar-outlines-llguidance-latency-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "structured-output-json-schema-xgrammar-outlines-llguidance-latency-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "structured-output-json-schema-xgrammar-outlines-llguidance-latency-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "structured-output-json-schema-xgrammar-outlines-llguidance-latency-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
