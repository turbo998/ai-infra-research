from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.in_network_computing_sharp_nvls_collective.run import (
    evaluate_in_network_collective_tiers,
)


class TestInNetworkComputingSharpNvlsCollective(unittest.TestCase):
    def test_provenance_and_sharp_invariants(self) -> None:
        data = evaluate_in_network_collective_tiers()
        self.assertEqual(
            data["experiment_id"], "in_network_computing_sharp_nvls_collective"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier3_vs_tier1_algbw_gain_x"], 1.84)
        self.assertEqual(summary["tier3_vs_tier1_latency_speedup_x"], 4.62)
        self.assertEqual(summary["tier3_network_traffic_reduction_pct"], 50.0)
        self.assertEqual(summary["tier3_sm_saved_per_gpu"], 16)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "in_network_computing_sharp_nvls_collective"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_in_network_collective_tiers())


if __name__ == "__main__":
    unittest.main()
