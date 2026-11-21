import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMFaultTolerantTrainingAsyncCheckpointTorchFTSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_fault_tolerant_training_async_checkpoint_torchft_local_replica_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_fault_tolerant_training_async_checkpoint_torchft_local_replica_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_TorchFT_Lighthouse_Replica_Isolation_Plus_P2P_RAM"]["Effective_Training_Time_Ratio_ETTR_Pct"],
            tiers["Tier_1_Sync_Shared_FS_Checkpoint_Cold_Restart"]["Effective_Training_Time_Ratio_ETTR_Pct"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-fault-tolerant-training-async-checkpoint-torchft-local-replica-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-fault-tolerant-training-async-checkpoint-torchft-local-replica-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-fault-tolerant-training-async-checkpoint-torchft-local-replica-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-fault-tolerant-training-async-checkpoint-torchft-local-replica-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-fault-tolerant-training-async-checkpoint-torchft-local-replica-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
