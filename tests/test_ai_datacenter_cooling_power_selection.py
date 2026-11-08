import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.ai_datacenter_cooling_power_selection.run import RESULTS_PATH, compute_results


class TestAiDatacenterCoolingPowerSelection(unittest.TestCase):
    def test_deterministic_model_and_check_flag(self) -> None:
        script = Path(__file__).resolve().parent.parent / "experiments" / "ai_datacenter_cooling_power_selection" / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data, compute_results())
        self.assertEqual(data["metadata"]["provenance_type"], "synthetic")
        self.assertFalse(data["metadata"]["independently_reproduced"])
        self.assertEqual(data["metadata"]["physical_datacenter_measurement"], "not_measured")

    def test_cooling_tiers_and_copper_nvlink_tco_invariants(self) -> None:
        data = compute_results()
        t1 = data["tiers"]["tier1_forced_air_aisle_containment"]
        t2 = data["tiers"]["tier2_rear_door_heat_exchanger_rdhx"]
        t3 = data["tiers"]["tier3_single_phase_direct_to_chip_dlc"]
        t4 = data["tiers"]["tier4_two_phase_or_immersion_cooling"]

        # Rack density & footprint invariants for 1,152 GPUs
        self.assertEqual(t1["racks_required"], 64)
        self.assertEqual(t2["racks_required"], 32)
        self.assertEqual(t3["racks_required"], 16)
        self.assertEqual(t3["rack_footprint_reduction_factor"], 4.0)

        # PUE monotonic improvement across cooling tiers
        self.assertEqual(t1["pue"], 1.48)
        self.assertEqual(t2["pue"], 1.28)
        self.assertEqual(t3["pue"], 1.12)
        self.assertEqual(t4["pue"], 1.05)

        # Tier 3 1-Phase DLC unlocks single-rack NVL72 passive copper backplane ($0 inter-rack scale-up optics)
        self.assertEqual(t1["inter_rack_scale_up_optics_bom_usd"], 4480000.00)
        self.assertEqual(t3["inter_rack_scale_up_optics_bom_usd"], 0.00)
        self.assertEqual(t3["power_savings_3yr_vs_air_usd"], 3148764.48)
        self.assertEqual(t3["total_tco_savings_3yr_vs_air_usd"], 6588764.48)
        self.assertEqual(t3["total_tco_savings_pct_vs_air"], 46.0)

        # Tier 4 Immersion has lower power cost than Tier 3 DLC, but higher 3-year total infra TCO due to tank/fluid CapEx and 2.5x slower MTTR
        self.assertLess(t4["power_cost_3yr_usd"], t3["power_cost_3yr_usd"])
        self.assertGreater(t4["total_3yr_infra_tco_usd"], t3["total_3yr_infra_tco_usd"])
        self.assertGreater(t4["serviceability_mttr_min"], t3["serviceability_mttr_min"] * 2.0)


if __name__ == "__main__":
    unittest.main()
