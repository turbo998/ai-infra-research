import unittest
from experiments.ai_cluster_rocev2_pfc_headroom_cable_length_xoff_xon_selection.run import (
    evaluate_rocev2_pfc_headroom_sizing_tiers,
)


class TestRocev2PfcHeadroomSizingSelection(unittest.TestCase):
    def test_pfc_headroom_tiers(self) -> None:
        data = evaluate_rocev2_pfc_headroom_sizing_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_48kb_headroom_overflow_drop_under_150m_fec"]
        t4 = tiers["tier4_cable_fec_calibrated_shared_headroom_pool_plus_dynamic_alpha"]
        self.assertGreater(t1["headroom_drop_rate_pct"], 1.0)
        self.assertEqual(t4["headroom_drop_rate_pct"], 0.0)
        self.assertGreater(t4["nccl_allreduce_busbw_gbps"], 370.0)


if __name__ == "__main__":
    unittest.main()
