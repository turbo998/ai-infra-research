import unittest
from experiments.ai_cluster_infiniband_subnet_manager_opensm_lid_reroute_flap_selection.run import (
    evaluate_infiniband_opensm_heavy_sweep_tiers,
)


class TestInfinibandOpensmHeavySweepSelection(unittest.TestCase):
    def test_infiniband_opensm_tiers(self) -> None:
        data = evaluate_infiniband_opensm_heavy_sweep_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_static_ftree_without_flap_suppression_global_heavy_sweep"]
        t4 = tiers["tier4_shield_ar_plus_flid_hierarchical_routing_and_flap_quarantine"]
        self.assertGreater(t1["failover_convergence_ms"], 4000.0)
        self.assertLess(t4["failover_convergence_ms"], 1.0)
        self.assertLess(t4["training_step_degradation_pct"], 0.5)


if __name__ == "__main__":
    unittest.main()
