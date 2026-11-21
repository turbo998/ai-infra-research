from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestDistributedCheckpointingPyTorchDcpMegatronAsyncOrbaxStorageSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "distributed_checkpointing_pytorch_dcp_megatron_async_orbax_storage_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["gpu_blocking_stall_sec"], 0.5)
        self.assertGreater(data["tiers"][3]["checkpoint_goodput_pct"], 99.95)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "distributed-checkpointing-pytorch-dcp-megatron-async-orbax-storage-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "distributed-checkpointing-pytorch-dcp-megatron-async-orbax-storage-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "distributed-checkpointing-pytorch-dcp-megatron-async-orbax-storage-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "distributed-checkpointing-pytorch-dcp-megatron-async-orbax-storage-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "distributed-checkpointing-pytorch-dcp-megatron-async-orbax-storage-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
