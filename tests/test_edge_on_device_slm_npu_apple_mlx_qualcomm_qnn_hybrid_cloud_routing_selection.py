import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEdgeOnDeviceSLMNPUHybridCloudRoutingSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "edge_on_device_slm_npu_apple_mlx_qualcomm_qnn_hybrid_cloud_routing_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "edge_on_device_slm_npu_apple_mlx_qualcomm_qnn_hybrid_cloud_routing_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Hybrid_Edge_Cloud_Confidence_Router_Plus_PII_Scrub"]["Monthly_Cloud_Inference_TCO_USD"],
            tiers["Tier_1_100Pct_Cloud_Frontier_LLM_No_Edge"]["Monthly_Cloud_Inference_TCO_USD"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "edge-on-device-slm-npu-apple-mlx-qualcomm-qnn-hybrid-cloud-routing-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "edge-on-device-slm-npu-apple-mlx-qualcomm-qnn-hybrid-cloud-routing-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "edge-on-device-slm-npu-apple-mlx-qualcomm-qnn-hybrid-cloud-routing-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "edge-on-device-slm-npu-apple-mlx-qualcomm-qnn-hybrid-cloud-routing-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "edge-on-device-slm-npu-apple-mlx-qualcomm-qnn-hybrid-cloud-routing-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
