from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmRuntimeWeightHotReloadRlNcclBroadcastCudaIpcSyncSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "llm_runtime_weight_hot_reload_rl_nccl_broadcast_cuda_ipc_sync_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["weight_sync_latency_sec"], 1.0)
        self.assertGreater(data["tiers"][3]["rl_step_goodput_pct"], 98.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "llm-runtime-weight-hot-reload-rl-nccl-broadcast-cuda-ipc-sync-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "llm-runtime-weight-hot-reload-rl-nccl-broadcast-cuda-ipc-sync-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "llm-runtime-weight-hot-reload-rl-nccl-broadcast-cuda-ipc-sync-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "llm-runtime-weight-hot-reload-rl-nccl-broadcast-cuda-ipc-sync-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "llm-runtime-weight-hot-reload-rl-nccl-broadcast-cuda-ipc-sync-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
