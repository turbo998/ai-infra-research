"""Unit tests for Round 174 AI Cluster L3 Routing Control Plane, BGP Unnumbered & RoCEv2 Multi-Rail Underlay Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_cluster_bgp_evpn_rocev2_rail_underlay_selection.run import (
    evaluate_ai_cluster_bgp_evpn_rocev2_rail_underlay_matrix,
)


class TestAiClusterBgpEvpnRocev2RailUnderlaySelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_cluster_bgp_evpn_rocev2_rail_underlay_matrix()
        self.assertEqual(data["experiment_id"], "ai_cluster_bgp_evpn_rocev2_rail_underlay_selection")
        self.assertEqual(data["round"], 174)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertEqual(
            tiers["tier3_rfc5549_bgp_unnumbered_rail_asn_soft_bfd"]["manual_ip_lines_4096_links"],
            0,
        )
        self.assertLess(
            tiers["tier4_bgp_unnumbered_hw_bfd_15ms_wcmp_evpn_vrf"]["failover_convergence_ms"],
            25.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_asymmetric_bw_gain_pts"],
            30.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_cluster_bgp_evpn_rocev2_rail_underlay_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_cluster_bgp_evpn_rocev2_rail_underlay_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
