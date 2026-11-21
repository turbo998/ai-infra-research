import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLLMKVCacheCompressionFP8KIVIDuoAttentionSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_kv_cache_compression_quantization_fp8_kivi_int2_snapkv_h2o_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "llm_kv_cache_compression_quantization_fp8_kivi_int2_snapkv_h2o_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertLess(
            tiers["Tier_4_Asymmetric_KIVI_INT4_Plus_DuoAttention_Retrieval_Heads"]["KV_Memory_Per_128K_Request_GB"],
            tiers["Tier_1_Uncompressed_BF16_KV_Cache"]["KV_Memory_Per_128K_Request_GB"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-kv-cache-compression-quantization-fp8-kivi-int2-snapkv-h2o-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "llm-kv-cache-compression-quantization-fp8-kivi-int2-snapkv-h2o-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "llm-kv-cache-compression-quantization-fp8-kivi-int2-snapkv-h2o-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-kv-cache-compression-quantization-fp8-kivi-int2-snapkv-h2o-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "llm-kv-cache-compression-quantization-fp8-kivi-int2-snapkv-h2o-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
