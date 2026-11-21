import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestKubernetesGatewayAPIInferenceExtensionGIESelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "kubernetes_gateway_api_inference_extension_gie_epp_kv_lora_routing_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "kubernetes_gateway_api_inference_extension_gie_epp_kv_lora_routing_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_K8s_Gateway_API_Inference_Extension_GIE_EPP"]["Cluster_P99_TTFT_ms"],
            tiers["Tier_1_K8s_ClusterIP_Nginx_Round_Robin"]["Cluster_P99_TTFT_ms"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "kubernetes-gateway-api-inference-extension-gie-epp-kv-lora-routing-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "kubernetes-gateway-api-inference-extension-gie-epp-kv-lora-routing-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "kubernetes-gateway-api-inference-extension-gie-epp-kv-lora-routing-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "kubernetes-gateway-api-inference-extension-gie-epp-kv-lora-routing-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "kubernetes-gateway-api-inference-extension-gie-epp-kv-lora-routing-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
