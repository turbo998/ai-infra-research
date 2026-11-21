import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmInferenceAccelerationPanoramaSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "llm_inference_acceleration_tp_pp_dp_ep_speculative_prefix_fp8_selection"
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
        t1 = tiers["tier1_cross_node_tp16_bf16_dense_style_on_moe"]
        t4 = tiers["tier4_fp8_dp_ep64_deepep_eplb_mtp_radix_prefix_cache"]
        self.assertLess(t4["p99_tpot_ms"], 12.0)
        self.assertGreater(t4["cluster_system_output_tps"], t1["cluster_system_output_tps"] * 5.0)
        self.assertLess(t4["cost_per_1m_output_tokens_usd"], t1["cost_per_1m_output_tokens_usd"] * 0.25)

    def test_artifacts_exist(self) -> None:
        slug = "llm-inference-acceleration-tp-pp-dp-ep-speculative-prefix-fp8"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
