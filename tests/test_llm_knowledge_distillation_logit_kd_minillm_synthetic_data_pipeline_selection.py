import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMKnowledgeDistillationLogitKDGKDSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_knowledge_distillation_logit_kd_minillm_synthetic_data_pipeline_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_knowledge_distillation_logit_kd_minillm_synthetic_data_pipeline_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_Verifier_Rejection_Sampling_Plus_OnPolicy_GKD_Reverse_KL"][
                "Student_8B_Vertical_Accuracy_Pct"
            ],
            tiers["Tier_1_Unfiltered_BlackBox_SeqKD_Hard_Label_SFT"][
                "Student_8B_Vertical_Accuracy_Pct"
            ],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-knowledge-distillation-logit-kd-minillm-synthetic-data-pipeline-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-knowledge-distillation-logit-kd-minillm-synthetic-data-pipeline-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-knowledge-distillation-logit-kd-minillm-synthetic-data-pipeline-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-knowledge-distillation-logit-kd-minillm-synthetic-data-pipeline-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-knowledge-distillation-logit-kd-minillm-synthetic-data-pipeline-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
