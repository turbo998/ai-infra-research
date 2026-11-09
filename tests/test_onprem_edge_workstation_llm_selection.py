from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.onprem_edge_workstation_llm_selection.run import RESULTS_PATH, compute_model


class TestOnpremEdgeWorkstationLlmSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_onprem_hardware_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_consumer_4x_rtx4090_rig"]
        t2 = tiers["tier2_unified_memory_soc_workstation"]
        t3 = tiers["tier3_dual_rtx_pro_6000_blackwell"]
        t4 = tiers["tier4_enterprise_2x_h200_nvl"]

        self.assertEqual(t1["annual_failure_rate_pct"], 14.5)
        self.assertEqual(t2["p99_16k_prefill_ttft_ms"], 4850.0)
        self.assertEqual(t3["vram_capacity_gb"], 192)
        self.assertEqual(t3["bs16_throughput_tok_s"], 740.0)
        self.assertEqual(t4["vram_capacity_gb"], 282)
        self.assertEqual(t4["p99_16k_prefill_ttft_ms"], 115.0)
        self.assertEqual(t4["bs16_throughput_tok_s"], 1680.0)


if __name__ == "__main__":
    unittest.main()
