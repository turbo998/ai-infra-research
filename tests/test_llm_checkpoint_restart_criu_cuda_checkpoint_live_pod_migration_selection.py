import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMCheckpointRestartCRIUCUDACheckpointSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_checkpoint_restart_criu_cuda_checkpoint_live_pod_migration_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_checkpoint_restart_criu_cuda_checkpoint_live_pod_migration_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_CRIU_Plus_NVIDIA_CUDA_Checkpoint_API_RDMA_Restore"]["Pod_Wakeup_To_First_Token_sec"],
            tiers["Tier_1_Cold_Pod_Boot_Full_Init_And_CUDA_Graph_Compile"]["Pod_Wakeup_To_First_Token_sec"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-checkpoint-restart-criu-cuda-checkpoint-live-pod-migration-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-checkpoint-restart-criu-cuda-checkpoint-live-pod-migration-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-checkpoint-restart-criu-cuda-checkpoint-live-pod-migration-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-checkpoint-restart-criu-cuda-checkpoint-live-pod-migration-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-checkpoint-restart-criu-cuda-checkpoint-live-pod-migration-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
