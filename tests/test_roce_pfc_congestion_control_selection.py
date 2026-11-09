"""Unit tests for Round 43 AI Cluster RoCEv2 PFC Storm & Congestion Control Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.roce_pfc_congestion_control_selection.run import (
    evaluate_roce_pfc_congestion_control_matrix,
)


class TestRocePfcCongestionControlSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_roce_pfc_congestion_control_matrix()
        self.assertEqual(data["experiment_id"], "roce_pfc_congestion_control_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier3_dcqcn_ecn_pfc_watchdog_rail_optimized"]["allreduce_busbw_gbps"],
            tiers["tier1_lossy_rocev2_static_ecmp_go_back_n"]["allreduce_busbw_gbps"] * 3.0,
        )
        self.assertGreater(
            tiers["tier4_spectrumx_uet_adaptive_packet_spraying"]["moe_alltoall_busbw_gbps"],
            tiers["tier2_pure_pfc_no_ecn_pause_storm"]["moe_alltoall_busbw_gbps"] * 2.0,
        )
        self.assertEqual(
            tiers["tier4_spectrumx_uet_adaptive_packet_spraying"]["pfc_storm_outages_per_month"],
            0.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "roce_pfc_congestion_control_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_roce_pfc_congestion_control_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
