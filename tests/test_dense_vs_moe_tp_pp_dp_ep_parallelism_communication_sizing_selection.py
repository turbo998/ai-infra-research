from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestDenseVsMoeTpPpDpEpParallelismCommunicationSizingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "dense_vs_moe_tp_pp_dp_ep_parallelism_communication_sizing_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["goodput_tok_s_per_gpu"], 2000)
        self.assertLess(data["tiers"][3]["exposed_comm_latency_ms_per_tok"], 1.0)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "dense-vs-moe-tp-pp-dp-ep-parallelism-communication-sizing-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "dense-vs-moe-tp-pp-dp-ep-parallelism-communication-sizing-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "dense-vs-moe-tp-pp-dp-ep-parallelism-communication-sizing-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "parallelism-tp-pp-dp-ep-moe-sandbox.html"
        sizer = self.repo_root / "docs" / "reports" / "dense-vs-moe-tp-pp-dp-ep-parallelism-communication-sizing-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "dense-vs-moe-tp-pp-dp-ep-parallelism-communication-sizing-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
