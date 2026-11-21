import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestVideoGenerationDiTWan21HunyuanUSPTeaCacheSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "video_generation_dit_wan21_hunyuan_usp_ulysses_ring_fp8_tea_cache_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "video_generation_dit_wan21_hunyuan_usp_ulysses_ring_fp8_tea_cache_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_xDiT_USP_Plus_FP8_SageAttention_TeaCache"]["Video_Generation_Latency_sec"],
            tiers["Tier_2_Pure_Tensor_Parallel_TP8_BF16"]["Video_Generation_Latency_sec"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "video-generation-dit-wan21-hunyuan-usp-ulysses-ring-fp8-tea-cache-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "video-generation-dit-wan21-hunyuan-usp-ulysses-ring-fp8-tea-cache-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "video-generation-dit-wan21-hunyuan-usp-ulysses-ring-fp8-tea-cache-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "video-generation-dit-wan21-hunyuan-usp-ulysses-ring-fp8-tea-cache-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "video-generation-dit-wan21-hunyuan-usp-ulysses-ring-fp8-tea-cache-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
