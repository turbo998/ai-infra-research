from __future__ import annotations

import unittest

from experiments.confidential_gpu_overhead.run import load_input, run_experiment


class ConfidentialGpuOverheadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_input()
        self.results = run_experiment(self.spec)

    def test_rim_verification_rules(self) -> None:
        by_id = {row["case_id"]: row for row in self.results["rim_verification_results"]}
        self.assertTrue(by_id["valid_hopper_nvdec0_disabled"]["verified"])
        self.assertTrue(by_id["valid_hopper_nvdec0_disabled"]["skipped_msr_35"])
        self.assertFalse(by_id["driver_vbios_index_conflict"]["verified"])
        self.assertIn("InvalidMeasurementIndexError", by_id["driver_vbios_index_conflict"]["failure_reason"])
        self.assertFalse(by_id["tampered_driver_measurement_mismatch"]["verified"])

    def test_ppcie_topology_fail_closed_on_rogue_switch(self) -> None:
        by_id = {row["case_id"]: row for row in self.results["ppcie_topology_results"]}
        self.assertTrue(by_id["valid_8gpu_4nvswitch_ppcie_mesh"]["topology_valid"])
        self.assertTrue(by_id["valid_8gpu_4nvswitch_ppcie_mesh"]["gpu_ready_state_enabled"])
        self.assertFalse(by_id["rogue_nvswitch_pdi_injection"]["topology_valid"])
        self.assertFalse(by_id["rogue_nvswitch_pdi_injection"]["gpu_ready_state_enabled"])

    def test_ppcie_restores_multi_gpu_tpot_while_h2d_stays_bounce_buffered(self) -> None:
        by_id = {row["id"]: row for row in self.results["dataplane_overhead_comparison"]}
        self.assertGreater(by_id["single_gpu_cc_on_sw_aes"]["tpot_slowdown_vs_non_cc"], 2.0)
        self.assertAlmostEqual(by_id["multi_gpu_ppcie_nvswitch"]["tpot_slowdown_vs_non_cc"], 1.0011, places=3)
        self.assertGreater(by_id["multi_gpu_ppcie_nvswitch"]["ttft_slowdown_vs_non_cc"], 1.35)


if __name__ == "__main__":
    unittest.main()
