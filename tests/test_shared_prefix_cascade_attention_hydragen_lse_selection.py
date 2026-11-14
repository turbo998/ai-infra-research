import unittest
from experiments.shared_prefix_cascade_attention_hydragen_lse_selection.run import (
    evaluate_cascade_attention_tiers,
)


class TestSharedPrefixCascadeAttentionSelection(unittest.TestCase):
    def test_cascade_attention_tiers(self) -> None:
        data = evaluate_cascade_attention_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t2 = tiers["tier2_standard_paged_attention_shared_page_table"]
        t4 = tiers["tier4_flashinfer_cascade_multi_depth_radix_lse_merge"]
        self.assertEqual(t2["shared_kv_hbm_reads_per_step"], 64)
        self.assertEqual(t4["shared_kv_hbm_reads_per_step"], 1)
        self.assertGreater(data["workload"]["speedup_over_paged_x"], 12.0)


if __name__ == "__main__":
    unittest.main()
