from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.ai_cluster_observability_sdc_selection.run import RESULTS_PATH, compute_model


class TestAiClusterObservabilitySdcSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_observability_sdc_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_basic_prometheus_polling"]
        t2 = tiers["tier2_dcgm_xid_auto_cordon"]
        t3 = tiers["tier3_flightrecorder_nccl_ras"]
        t4 = tiers["tier4_online_sdc_golden_verifier"]

        self.assertEqual(t1["root_cause_triage_sec"], 8700.0)
        self.assertEqual(t1["effective_goodput_pct"], 68.4)
        self.assertEqual(t2["effective_goodput_pct"], 82.2)
        self.assertEqual(t3["root_cause_triage_sec"], 0.65)
        self.assertEqual(t3["effective_goodput_pct"], 91.5)
        self.assertEqual(t4["sdc_detection_rate_pct"], 99.6)
        self.assertEqual(t4["effective_goodput_pct"], 96.4)
        self.assertEqual(t4["monthly_savings_vs_tier1_usd"], 1046528.00)
        self.assertEqual(t4["waste_reduction_pct"], 88.6)


if __name__ == "__main__":
    unittest.main()
