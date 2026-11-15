import unittest
from experiments.gpu_pcie_gen5_retimer_signal_integrity_link_downgrade_xid79_selection.run import evaluate_pcie_gen5_link_downgrade_and_margining


class TestPcieGen5LinkDowngradeAndMargining(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_pcie_gen5_link_downgrade_and_margining()
        self.assertEqual(res["round"], 284)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier4_lnksta_gate_plus_ltssm_aer_counter_plus_gen5_lane_margining"]["effective_gdr_bandwidth_gbps"],
            tiers["tier1_unmonitored_silent_pcie_x4_gen4_downgrade"]["effective_gdr_bandwidth_gbps"] * 7.5,
        )
        self.assertGreater(
            tiers["tier4_lnksta_gate_plus_ltssm_aer_counter_plus_gen5_lane_margining"]["cluster_allreduce_efficiency_pct"],
            98.0,
        )


if __name__ == "__main__":
    unittest.main()
