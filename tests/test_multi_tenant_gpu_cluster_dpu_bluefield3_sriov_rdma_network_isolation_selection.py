from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestMultiTenantGpuClusterDpuBluefield3SriovRdmaNetworkIsolationSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "multi_tenant_gpu_cluster_dpu_bluefield3_sriov_rdma_network_isolation_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["noisy_neighbor_victim_busbw_drop_pct"], 1.0)
        self.assertEqual(data["tiers"][3]["host_cpu_cores_consumed_by_sdn"], 0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "multi-tenant-gpu-cluster-dpu-bluefield3-sriov-rdma-network-isolation-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "multi-tenant-gpu-cluster-dpu-bluefield3-sriov-rdma-network-isolation-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "multi-tenant-gpu-cluster-dpu-bluefield3-sriov-rdma-network-isolation-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "multi-tenant-gpu-cluster-dpu-bluefield3-sriov-rdma-network-isolation-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "multi-tenant-gpu-cluster-dpu-bluefield3-sriov-rdma-network-isolation-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
