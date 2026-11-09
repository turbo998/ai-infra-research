from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.nvlink_pcie_cxl_scaleup_selection.run import RESULTS_PATH, compute_model


class TestNvlinkPcieCxlScaleupSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_scaleup_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_pcie_gen5_switch_tree"]
        t2 = tiers["tier2_hgx_nvswitch_8gpu"]
        t3 = tiers["tier3_nvl72_mnnvl_copper_rack"]
        t4 = tiers["tier4_cxl3_memory_pool_hybrid"]

        self.assertEqual(t1["tpot_ms"], 20.20)
        self.assertFalse(t1["sla_12ms_pass"])
        self.assertEqual(t2["tpot_ms"], 9.35)
        self.assertTrue(t2["sla_12ms_pass"])
        self.assertEqual(t2["throughput_speedup_x"], 2.16)
        self.assertEqual(t3["tpot_ms"], 5.40)
        self.assertTrue(t3["sla_12ms_pass"])
        self.assertEqual(t3["throughput_speedup_x"], 4.49)
        self.assertEqual(t3["cost_savings_pct"], 62.3)
        self.assertEqual(t4["kv_128k_reload_ttft_ms"], 18.5)
        self.assertEqual(t4["cost_savings_pct"], 56.5)


if __name__ == "__main__":
    unittest.main()
