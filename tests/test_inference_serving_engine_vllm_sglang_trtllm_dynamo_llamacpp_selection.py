import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestInferenceServingEngineVllmSglangTrtllmDynamoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "inference_serving_engine_vllm_sglang_trtllm_dynamo_llamacpp_selection"
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
        t2 = tiers["tier2_vllm_v1_unified_engine_general_enterprise_maas_standard"]
        t4 = tiers["tier4_dynamo_kv_router_nixl_orchestrating_sglang_vllm_trtllm_workers"]
        self.assertGreater(t4["sla_compliant_goodput_tps_per_gpu"], t2["sla_compliant_goodput_tps_per_gpu"])
        self.assertGreater(t4["multi_turn_prefix_cache_hit_rate_pct"], 90.0)

    def test_artifacts_exist(self) -> None:
        slug = "inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
