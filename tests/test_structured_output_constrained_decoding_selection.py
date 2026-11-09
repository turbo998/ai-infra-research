from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.structured_output_constrained_decoding_selection.run import RESULTS_PATH, compute_model


class TestStructuredOutputConstrainedDecodingSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_constrained_decoding_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_prompt_retry_unconstrained"]
        t2 = tiers["tier2_naive_per_token_regex_scan"]
        t3 = tiers["tier3_outlines_precompiled_fsm"]
        t4 = tiers["tier4_xgrammar_async_jump_forward"]

        self.assertEqual(t1["schema_valid_rate_pct"], 84.5)
        self.assertEqual(t2["mask_cpu_overhead_ms"], 18.50)
        self.assertEqual(t3["schema_valid_rate_pct"], 100.0)
        self.assertEqual(t3["mask_cpu_overhead_ms"], 0.45)
        self.assertEqual(t4["schema_valid_rate_pct"], 100.0)
        self.assertEqual(t4["effective_tpot_ms"], 6.10)
        self.assertEqual(t4["speedup_vs_unconstrained_x"], 1.56)
        self.assertEqual(t4["cost_savings_vs_tier1_pct"], 51.2)


if __name__ == "__main__":
    unittest.main()
