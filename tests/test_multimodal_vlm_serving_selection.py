import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "multimodal_vlm_serving_selection" / "run.py"
RESULTS_PATH = REPO_ROOT / "experiments" / "multimodal_vlm_serving_selection" / "results.json"


class TestMultimodalVlmServingSelection(unittest.TestCase):
    def test_check_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_invariants_and_provenance(self) -> None:
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["round"], 13)
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_monolithic_colocated_vlm"]
        t2 = tiers["tier2_gpu_nvdec_epd_disaggregation"]
        t3 = tiers["tier3_epd_plus_vision_embedding_cache"]
        t4 = tiers["tier4_epd_cache_plus_adaptive_token_compression"]

        self.assertGreater(t1["p99_tpot_ms"], 300.0)
        self.assertLessEqual(t2["p99_tpot_ms"], 50.0)
        self.assertLess(t3["avg_ttft_ms"], t2["avg_ttft_ms"])
        self.assertLess(t4["avg_ttft_ms"], t3["avg_ttft_ms"])
        self.assertEqual(t1["kv_cache_per_request_gb"], 5.41)
        self.assertEqual(t4["kv_cache_per_request_gb"], 1.47)
        self.assertEqual(t4["cluster_throughput_req_per_s"], 24.6)
        self.assertEqual(t4["cost_per_1k_requests_usd"], 0.45)


if __name__ == "__main__":
    unittest.main()
