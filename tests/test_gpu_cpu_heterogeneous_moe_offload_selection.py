from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.gpu_cpu_heterogeneous_moe_offload_selection.run import (
    evaluate_heterogeneous_offload_tiers,
)


class TestGpuCpuHeterogeneousMoeOffloadSelection(unittest.TestCase):
    def test_provenance_and_offload_invariants(self) -> None:
        data = evaluate_heterogeneous_offload_tiers()
        self.assertEqual(
            data["experiment_id"], "gpu_cpu_heterogeneous_moe_offload_selection"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier2_interactive_speedup_x"], 8.86)
        self.assertEqual(summary["tier4_interactive_decode_tps"], 18.6)
        self.assertEqual(summary["tier4_pcie_payload_reduction_x"], 531250.0)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "gpu_cpu_heterogeneous_moe_offload_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_heterogeneous_offload_tiers())


if __name__ == "__main__":
    unittest.main()
