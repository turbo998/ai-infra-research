from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.confidential_ai_tee_selection.run import RESULTS_PATH, compute_model


class TestConfidentialAiTeeSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_confidential_tee_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_plaintext_gpu_container"]
        t2 = tiers["tier2_pii_redaction_gateway"]
        t3 = tiers["tier3_single_gpu_cc_on_tee"]
        t4 = tiers["tier4_multigpu_ppcie_nvlink_tee"]

        self.assertEqual(t1["latency_overhead_pct"], 0.0)
        self.assertEqual(t2["residual_pii_leakage_pct"], 18.5)
        self.assertEqual(t2["accuracy_drop_pct"], 6.4)
        self.assertEqual(t3["latency_overhead_pct"], 4.8)
        self.assertEqual(t3["residual_pii_leakage_pct"], 0.0)
        self.assertEqual(t4["latency_overhead_pct"], 6.2)
        self.assertEqual(t4["naive_cc_on_tp8_tpot_ms"], 42.60)
        self.assertEqual(t4["tpot_ms"], 11.90)


if __name__ == "__main__":
    unittest.main()
