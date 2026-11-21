import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestModelMergingKitMergeKitTIESDARESelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "model_merging_kit_mergekit_ties_dare_slerp_frankenmoe_routing_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "model_merging_kit_mergekit_ties_dare_slerp_frankenmoe_routing_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_MergeKit_DARE_TIES_DELLA_Plus_Layerwise_SLERP"]["Composite_4Domain_Benchmark_Score_Pct"],
            tiers["Tier_3_Naive_Linear_Average_Task_Arithmetic"]["Composite_4Domain_Benchmark_Score_Pct"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "model-merging-kit-mergekit-ties-dare-slerp-frankenmoe-routing-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "model-merging-kit-mergekit-ties-dare-slerp-frankenmoe-routing-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "model-merging-kit-mergekit-ties-dare-slerp-frankenmoe-routing-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "model-merging-kit-mergekit-ties-dare-slerp-frankenmoe-routing-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "model-merging-kit-mergekit-ties-dare-slerp-frankenmoe-routing-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
