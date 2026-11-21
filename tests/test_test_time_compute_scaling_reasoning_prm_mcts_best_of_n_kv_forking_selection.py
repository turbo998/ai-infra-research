import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestTestTimeComputeScalingReasoningPRMMCTSSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "test_time_compute_scaling_reasoning_prm_mcts_best_of_n_kv_forking_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "test_time_compute_scaling_reasoning_prm_mcts_best_of_n_kv_forking_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Adaptive_Difficulty_Router_PRM_Beam_KV_Forking"]["Cost_Per_1K_Solved_Tasks_USD"],
            tiers["Tier_3_Shared_Prefix_Best_Of_16_Plus_ORM"]["Cost_Per_1K_Solved_Tasks_USD"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "test-time-compute-scaling-reasoning-prm-mcts-best-of-n-kv-forking-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "test-time-compute-scaling-reasoning-prm-mcts-best-of-n-kv-forking-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "test-time-compute-scaling-reasoning-prm-mcts-best-of-n-kv-forking-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "test-time-compute-scaling-reasoning-prm-mcts-best-of-n-kv-forking-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "test-time-compute-scaling-reasoning-prm-mcts-best-of-n-kv-forking-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
