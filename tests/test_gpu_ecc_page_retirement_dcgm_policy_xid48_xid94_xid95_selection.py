import unittest
from experiments.gpu_ecc_page_retirement_dcgm_policy_xid48_xid94_xid95_selection.run import (
    evaluate_xid_containment_remediation_tiers,
)


class TestGpuXidContainmentRemediationSelection(unittest.TestCase):
    def test_xid_containment_tiers(self) -> None:
        data = evaluate_xid_containment_remediation_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_blanket_cordon_on_any_xid_including_xid31_xid43_user_bugs"]
        t4 = tiers["tier4_three_class_xid_matrix_xid31_ignore_xid94_inband_reset_xid95_drain"]
        self.assertGreater(t1["false_node_cordon_rate_pct"], 80.0)
        self.assertEqual(t4["false_node_cordon_rate_pct"], 0.0)
        self.assertFalse(t4["uncontained_xid95_leak_risk"])
        self.assertLess(t4["mean_node_recovery_downtime_min"], 1.0)


if __name__ == "__main__":
    unittest.main()
