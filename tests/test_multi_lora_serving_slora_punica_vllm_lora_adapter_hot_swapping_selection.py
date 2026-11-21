import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestMultiLoRAServingSLoRAPunicavLLMSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "multi_lora_serving_slora_punica_vllm_lora_adapter_hot_swapping_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "multi_lora_serving_slora_punica_vllm_lora_adapter_hot_swapping_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_S_LoRA_Unified_Paging_Plus_Consistent_Hash_Gateway"]["Required_H200_GPUs"],
            tiers["Tier_1_Dedicated_Merged_Weight_Pods_Per_Tenant"]["Required_H200_GPUs"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "multi-lora-serving-slora-punica-vllm-lora-adapter-hot-swapping-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "multi-lora-serving-slora-punica-vllm-lora-adapter-hot-swapping-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "multi-lora-serving-slora-punica-vllm-lora-adapter-hot-swapping-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "multi-lora-serving-slora-punica-vllm-lora-adapter-hot-swapping-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "multi-lora-serving-slora-punica-vllm-lora-adapter-hot-swapping-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
