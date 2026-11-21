import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmQuantizationFp8BlockwiseNvfp4Selection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_quantization_fp8_blockwise_nvfp4_awq_marlin_accuracy_throughput_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_script_check_and_invariants(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"], "synthetic + anonymized_reported")
        self.assertFalse(data["independently_reproduced"])
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unquantized_bf16_weights_and_bf16_kv_cache"]
        t4 = tiers["tier4_fine_grained_block_fp8_1x128_128x128_and_blackwell_nvfp4"]
        self.assertGreaterEqual(t4["relative_benchmark_accuracy_retention_pct"], 99.5)
        self.assertGreater(
            t4["sla_compliant_goodput_tps_per_gpu"],
            t1["sla_compliant_goodput_tps_per_gpu"] * 2.0,
        )

    def test_artifacts_exist(self) -> None:
        slug = "llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
