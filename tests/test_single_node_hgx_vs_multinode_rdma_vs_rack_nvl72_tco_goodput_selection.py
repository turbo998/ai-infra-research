from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestSingleNodeHgxVsMultinodeRdmaVsRackNvl72TcoGoodputSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "single_node_hgx_vs_multinode_rdma_vs_rack_nvl72_tco_goodput_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertGreater(data["tiers"][3]["goodput_tok_s_per_gpu"], 7000)
        self.assertLess(data["tiers"][3]["cost_per_1m_tokens_usd"], 0.15)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "single-node-hgx-vs-multinode-rdma-vs-rack-nvl72-tco-goodput-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "single-node-hgx-vs-multinode-rdma-vs-rack-nvl72-tco-goodput-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "single-node-hgx-vs-multinode-rdma-vs-rack-nvl72-tco-goodput-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "hgx-vs-multinode-rdma-vs-nvl72-rack-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "single-node-hgx-vs-multinode-rdma-vs-rack-nvl72-tco-goodput-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "single-node-hgx-vs-multinode-rdma-vs-rack-nvl72-tco-goodput-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
