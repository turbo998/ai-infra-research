import unittest

from experiments.optical_cpo_ocs_fabric.run import (
    check_outputs,
    evaluate_architecture_at_scale,
    generate_results,
    load_input,
)


class OpticalCpoOcsFabricTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_input()
        self.results = generate_results(self.config)

    def test_outputs_in_sync(self) -> None:
        self.assertTrue(check_outputs(self.results))

    def test_cpo_and_ocs_reduce_power_and_improve_goodput(self) -> None:
        summary = self.results["summary"]
        self.assertAlmostEqual(summary["dsp_100k_optics_power_mw"], 7.92, places=2)
        self.assertAlmostEqual(summary["cpo_100k_optics_power_mw"], 2.16, places=2)
        self.assertAlmostEqual(summary["hybrid_100k_optics_power_mw"], 0.72, places=2)
        self.assertGreater(summary["cpo_100k_sync_goodput_pct"], summary["dsp_100k_sync_goodput_pct"] + 20.0)
        self.assertGreater(summary["hybrid_100k_sync_goodput_pct"], summary["cpo_100k_sync_goodput_pct"])

    def test_elsfp_blast_radius_captures_shared_ports(self) -> None:
        cpo_arch = next(a for a in self.config["architectures"] if a["arch_id"] == "cpo_elsfp_3tier_800g")
        row = evaluate_architecture_at_scale(cpo_arch, 100000, 800.0, 1800.0, 240.0)
        self.assertEqual(row["oe_shared_ports"], 2)
        self.assertEqual(row["elsfp_shared_ports"], 8)
        self.assertAlmostEqual(row["elsfp_blast_radius_gbps"], 6400.0, places=1)


if __name__ == "__main__":
    unittest.main()
