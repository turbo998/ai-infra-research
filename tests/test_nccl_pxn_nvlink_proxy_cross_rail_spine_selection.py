import unittest
from experiments.nccl_pxn_nvlink_proxy_cross_rail_spine_selection.run import (
    evaluate_nccl_pxn_tiers,
)


class TestNcclPxnCrossRailSelection(unittest.TestCase):
    def test_pxn_tiers(self) -> None:
        data = evaluate_nccl_pxn_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_pxn_disabled_cross_rail_spine_hairpin"]
        t3 = tiers["tier3_nccl_pxn_enabled_with_gdr_pix_and_p2p_level2"]
        t4 = tiers["tier4_hierarchical_nvlink_rail_aligned_deepep_pxn"]
        self.assertEqual(t1["spine_traffic_ratio_pct"], 87.5)
        self.assertEqual(t3["spine_traffic_ratio_pct"], 0.0)
        self.assertGreaterEqual(t4["per_gpu_busbw_gbs"], 48.0)


if __name__ == "__main__":
    unittest.main()
