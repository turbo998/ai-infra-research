from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestComputeCommunicationOverlapUserbuffersFsdp2NcclSmContentionSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "compute_communication_overlap_userbuffers_fsdp2_nccl_sm_contention_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLessEqual(data["tiers"][3]["sms_consumed_by_comm"], 2)
        self.assertGreater(data["tiers"][3]["effective_cluster_mfu_pct"], 56.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "compute-communication-overlap-userbuffers-fsdp2-nccl-sm-contention-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "compute-communication-overlap-userbuffers-fsdp2-nccl-sm-contention-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "compute-communication-overlap-userbuffers-fsdp2-nccl-sm-contention-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "compute-communication-overlap-userbuffers-fsdp2-nccl-sm-contention-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "compute-communication-overlap-userbuffers-fsdp2-nccl-sm-contention-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
