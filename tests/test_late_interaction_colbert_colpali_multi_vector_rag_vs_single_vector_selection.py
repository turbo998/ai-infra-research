import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLateInteractionColBERTColPaliMultiVectorSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "late_interaction_colbert_colpali_multi_vector_rag_vs_single_vector_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "late_interaction_colbert_colpali_multi_vector_rag_vs_single_vector_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_TwoStage_Pooled_SingleVector_Prefetch_Plus_Binary_MaxSim_Rescore"]["Index_Memory_Footprint_GB"],
            tiers["Tier_2_Uncompressed_ColPali_ColBERT_Full_Token_Vectors"]["Index_Memory_Footprint_GB"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "late-interaction-colbert-colpali-multi-vector-rag-vs-single-vector-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "late-interaction-colbert-colpali-multi-vector-rag-vs-single-vector-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "late-interaction-colbert-colpali-multi-vector-rag-vs-single-vector-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "late-interaction-colbert-colpali-multi-vector-rag-vs-single-vector-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "late-interaction-colbert-colpali-multi-vector-rag-vs-single-vector-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
