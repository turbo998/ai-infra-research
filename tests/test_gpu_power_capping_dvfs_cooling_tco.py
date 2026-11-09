from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.gpu_power_capping_dvfs_cooling_tco.run import (
    evaluate_gpu_power_cooling_tiers,
)


class TestGpuPowerCappingDvfsCoolingTco(unittest.TestCase):
    def test_provenance_and_power_efficiency_invariants(self) -> None:
        data = evaluate_gpu_power_cooling_tiers()
        self.assertEqual(data["experiment_id"], "gpu_power_capping_dvfs_cooling_tco")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(data["provenance"]["physical_gpu_cluster_measurement"], "not_measured")

        summary = data["summary"]
        self.assertEqual(summary["static_550w_power_saving_pct"], 21.4)
        self.assertEqual(summary["static_550w_blended_perf_loss_pct"], 3.4)
        self.assertEqual(summary["tier4_vs_tier1_efficiency_gain_x"], 1.56)
        self.assertEqual(summary["tier4_vs_tier1_3yr_opex_savings_usd"], 1647734)
        self.assertEqual(summary["tier4_extra_gpus_deployed_under_1_2mw"], 424)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "gpu_power_capping_dvfs_cooling_tco"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_gpu_power_cooling_tiers())


if __name__ == "__main__":
    unittest.main()
