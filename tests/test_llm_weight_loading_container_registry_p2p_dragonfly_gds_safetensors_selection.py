import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMWeightLoadingDragonflyP2PGDSSafetensorsSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_weight_loading_container_registry_p2p_dragonfly_gds_safetensors_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_weight_loading_container_registry_p2p_dragonfly_gds_safetensors_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Nydus_Lazy_P2P_Plus_Runai_GDS_RDMA_Weight_Broadcast"][
                "Total_64Node_Pod_Ready_Latency_sec"
            ],
            tiers["Tier_1_Baked_158GB_OCI_Image_Central_Registry_Pull"][
                "Total_64Node_Pod_Ready_Latency_sec"
            ],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-weight-loading-container-registry-p2p-dragonfly-gds-safetensors-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-weight-loading-container-registry-p2p-dragonfly-gds-safetensors-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-weight-loading-container-registry-p2p-dragonfly-gds-safetensors-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-weight-loading-container-registry-p2p-dragonfly-gds-safetensors-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-weight-loading-container-registry-p2p-dragonfly-gds-safetensors-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
