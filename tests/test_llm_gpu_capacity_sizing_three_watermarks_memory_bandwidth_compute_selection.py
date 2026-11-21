import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmGpuCapacitySizingThreeWatermarksSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_gpu_capacity_sizing_three_watermarks_memory_bandwidth_compute_selection"
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
        t1 = tiers["tier1_naive_weights_only_bf16_sizing_without_kv_or_bandwidth_check"]
        t4 = tiers["tier4_three_watermark_fp8_h200_b200_co_sizing_with_085_margin"]
        self.assertEqual(t4["cuda_oom_and_sla_breach_rate_pct"], 0.0)
        self.assertLess(t4["achieved_p99_tpot_ms"], 15.0)
        self.assertGreater(
            t4["max_concurrent_16k_sessions_per_8gpu_node"],
            t1["max_concurrent_16k_sessions_per_8gpu_node"] * 8,
        )

    def test_artifacts_exist(self) -> None:
        slug = "llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
