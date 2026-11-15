import unittest
from experiments.gpu_liquid_cooling_cdu_manifold_leak_inlet_temp_throttling_selection.run import evaluate_dlc_liquid_cooling_rack


class TestDlcLiquidCoolingRack(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_dlc_liquid_cooling_rack()
        self.assertEqual(res["round"], 281)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_42c_inlet_underflow_no_rdhx_air_assist"]["hbm3e_peak_temp_c"],
            85.0,
        )
        self.assertLess(
            tiers["tier4_32c_dlc_rdhx_plus_dewpoint_guard_plus_bmc_25s_dcp_drain"]["hbm3e_peak_temp_c"],
            75.0,
        )
        self.assertEqual(
            tiers["tier4_32c_dlc_rdhx_plus_dewpoint_guard_plus_bmc_25s_dcp_drain"]["cluster_step_throughput_pct"],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
