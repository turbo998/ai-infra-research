import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestKubernetesDRATopologyAwareGPUNICSchedulingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "kubernetes_dra_topology_aware_gpu_nic_numa_scheduling_kueue_volcano_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "kubernetes_dra_topology_aware_gpu_nic_numa_scheduling_kueue_volcano_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_K8s_DRA_PCIe_Switch_CoAffinity_Plus_Kueue_TAS"]["Effective_GPUDirect_RDMA_BW_GBps"],
            tiers["Tier_1_Legacy_Device_Plugin_Topology_None"]["Effective_GPUDirect_RDMA_BW_GBps"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
