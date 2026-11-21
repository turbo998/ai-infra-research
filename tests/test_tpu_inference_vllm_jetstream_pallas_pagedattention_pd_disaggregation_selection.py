import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestTpuInferenceVllmJetstreamSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "tpu_inference_vllm_jetstream_pallas_pagedattention_pd_disaggregation_selection"
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
        t1 = tiers["tier1_unbucketed_huggingface_tpu_contiguous_kv"]
        t4 = tiers["tier4_vllm_tpu_pallas_ragged_pa_ici_pd_disaggregation_prefix_cache"]
        self.assertLess(t4["p99_tpot_ms"], 16.0)
        self.assertGreater(t4["decode_hbm_bandwidth_utilization_mbu_pct"], 78.0)
        self.assertLess(t4["sla_compliant_cost_per_1m_tokens_usd"], 0.25)
        self.assertLess(
            t4["monthly_64_chip_serving_equivalent_tco_usd"],
            t1["monthly_64_chip_serving_equivalent_tco_usd"] * 0.30,
        )

    def test_artifacts_exist(self) -> None:
        slug = "tpu-inference-vllm-jetstream-pallas-pagedattention-pd-disaggregation"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
