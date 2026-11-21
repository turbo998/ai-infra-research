import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmCustomizationPromptRagSftDistillationSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_customization_prompt_rag_sft_lora_distillation_cascade_router_selection"
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
        t2 = tiers["tier2_100pct_frontier_teacher_model_plus_hybrid_rag"]
        t4 = tiers["tier4_raft_distilled_8b_lora_plus_hybrid_rag_and_82_18_cascade_router"]
        self.assertGreaterEqual(t4["domain_task_accuracy_pct"], t2["domain_task_accuracy_pct"])
        self.assertLess(t4["monthly_60m_queries_total_tco_usd"], t2["monthly_60m_queries_total_tco_usd"] * 0.25)

    def test_artifacts_exist(self) -> None:
        slug = "llm-customization-prompt-rag-sft-lora-distillation-cascade-router"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
