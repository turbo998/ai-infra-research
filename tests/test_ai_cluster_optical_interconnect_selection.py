import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "experiments" / "ai_cluster_optical_interconnect_selection" / "run.py"
RESULTS_PATH = REPO_ROOT / "experiments" / "ai_cluster_optical_interconnect_selection" / "results.json"


class TestAIClusterOpticalInterconnectSelection(unittest.TestCase):
    def test_deterministic_check_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)

    def test_optical_power_flap_and_tco_invariants(self) -> None:
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"]["provenance_type"], "synthetic")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(data["provenance"]["physical_datacenter_fabric_measurement"], "not_measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_all_dsp_pluggable_optics"]
        t2 = tiers["tier2_inrack_copper_plus_spine_dsp"]
        t3 = tiers["tier3_inrack_copper_plus_spine_lpo"]
        t4 = tiers["tier4_cpo_silicon_photonics_plus_mems_ocs"]

        # Power monotonically improves: Tier 1 (131.07 kW) > Tier 2 (75.78 kW) > Tier 3 (40.96 kW) > Tier 4 (20.89 kW)
        self.assertAlmostEqual(t1["total_interconnect_power_kw"], 131.07, places=2)
        self.assertAlmostEqual(t2["total_interconnect_power_kw"], 75.78, places=2)
        self.assertAlmostEqual(t3["total_interconnect_power_kw"], 40.96, places=2)
        self.assertAlmostEqual(t4["total_interconnect_power_kw"], 20.89, places=2)

        # Link flaps over 3 years decrease from 442.4 -> 239.6 -> 110.6 -> 36.9 (12.0x reduction in Tier 4)
        self.assertAlmostEqual(t1["expected_link_flaps_3yr"], 442.4, places=1)
        self.assertAlmostEqual(t2["expected_link_flaps_3yr"], 239.6, places=1)
        self.assertAlmostEqual(t3["expected_link_flaps_3yr"], 110.6, places=1)
        self.assertAlmostEqual(t4["expected_link_flaps_3yr"], 36.9, places=1)
        self.assertGreaterEqual(t4["flap_reduction_factor_x"], 11.9)

        # 3-Year TCO checks
        self.assertAlmostEqual(t1["total_3yr_tco_usd"], 9450699.46, delta=100.0)
        self.assertLess(t2["total_3yr_tco_usd"], t1["total_3yr_tco_usd"])
        self.assertLess(t3["total_3yr_tco_usd"], t2["total_3yr_tco_usd"])
        self.assertLess(t4["total_3yr_tco_usd"], t2["total_3yr_tco_usd"])


if __name__ == "__main__":
    unittest.main()
