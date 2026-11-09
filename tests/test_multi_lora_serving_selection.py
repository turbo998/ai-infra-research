from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.multi_lora_serving_selection.run import RESULTS_PATH, compute_model


class TestMultiLoraServingSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_multi_lora_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_dedicated_merged_replicas"]
        t2 = tiers["tier2_serial_adapter_hotswap"]
        t3 = tiers["tier3_vllm_fixed_slot_multi_lora"]
        t4 = tiers["tier4_slora_unified_paged_mbgmm"]

        self.assertEqual(t1["required_gpus"], 128)
        self.assertEqual(t1["monthly_cluster_usd"], 233600.00)
        self.assertEqual(t2["p99_ttft_ms"], 1840.0)
        self.assertFalse(t2["sla_pass"])
        self.assertEqual(t3["required_gpus"], 4)
        self.assertTrue(t3["sla_pass"])
        self.assertEqual(t4["required_gpus"], 4)
        self.assertEqual(t4["p99_ttft_ms"], 92.0)
        self.assertEqual(t4["p99_tpot_ms"], 12.4)
        self.assertEqual(t4["cluster_qps"], 264.0)
        self.assertEqual(t4["cost_savings_vs_tier1_pct"], 96.9)


if __name__ == "__main__":
    unittest.main()
