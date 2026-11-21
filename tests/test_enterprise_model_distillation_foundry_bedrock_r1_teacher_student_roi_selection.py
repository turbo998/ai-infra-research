from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseModelDistillationFoundryBedrockR1TeacherStudentRoiSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_model_distillation_foundry_bedrock_r1_teacher_student_roi_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["domain_accuracy_pct"], 95.0)
        self.assertLess(data["tiers"][3]["annual_serving_tco_usd"], 210000)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-model-distillation-foundry-bedrock-r1-teacher-student-roi-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-model-distillation-foundry-bedrock-r1-teacher-student-roi-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-model-distillation-foundry-bedrock-r1-teacher-student-roi-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "model-distillation-foundry-bedrock-r1-teacher-student-roi-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-model-distillation-foundry-bedrock-r1-teacher-student-roi-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-model-distillation-foundry-bedrock-r1-teacher-student-roi-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
