from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.multi_region_dr_cross_cloud_llm_serving.run import (
    evaluate_multi_region_dr_tiers,
)


class TestMultiRegionDrCrossCloudLlmServing(unittest.TestCase):
    def test_provenance_and_dr_invariants(self) -> None:
        data = evaluate_multi_region_dr_tiers()
        self.assertEqual(
            data["experiment_id"], "multi_region_dr_cross_cloud_llm_serving"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier2_kv_hit_rate_drop_pt"], 47.2)
        self.assertEqual(summary["tier4_vs_tier2_ttft_speedup_x"], 6.62)
        self.assertEqual(summary["tier4_failover_rto_sec"], 1.8)
        self.assertEqual(summary["tier4_vs_static_dual_region_tco_saving_pct"], 41.5)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "multi_region_dr_cross_cloud_llm_serving"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_multi_region_dr_tiers())


if __name__ == "__main__":
    unittest.main()
