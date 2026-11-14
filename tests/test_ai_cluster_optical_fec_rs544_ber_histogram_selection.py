import unittest
from experiments.ai_cluster_optical_fec_rs544_ber_histogram_selection.run import (
    evaluate_optical_fec_tiers,
)


class TestOpticalFecRs544Selection(unittest.TestCase):
    def test_optical_fec_tiers(self) -> None:
        data = evaluate_optical_fec_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(data["workload"]["max_correctable_symbols"], 15)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_binary_link_up_down_only"]
        t4 = tiers["tier4_rs544_fec_tail_bin12_15_plus_cmis_dom_auto_drain"]
        self.assertLess(t1["effective_allreduce_busbw_gbs"], 20.0)
        self.assertGreaterEqual(t4["effective_allreduce_busbw_gbs"], 48.0)
        self.assertGreaterEqual(t4["cluster_goodput_pct"], 99.0)


if __name__ == "__main__":
    unittest.main()
