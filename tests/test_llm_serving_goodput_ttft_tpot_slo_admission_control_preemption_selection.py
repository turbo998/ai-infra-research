import unittest
from experiments.llm_serving_goodput_ttft_tpot_slo_admission_control_preemption_selection.run import (
    evaluate_serving_goodput_preemption_tiers,
)


class TestServingGoodputPreemptionSelection(unittest.TestCase):
    def test_goodput_preemption_tiers(self) -> None:
        data = evaluate_serving_goodput_preemption_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unbounded_max_num_seqs_256_plus_recompute_preemption_thrash"]
        t4 = tiers["tier4_kv_watermark_admission_gate_82pct_plus_tpot_token_cap_and_async_pinned_swap"]
        self.assertLess(t1["joint_slo_goodput_compliance_pct"], 15.0)
        self.assertGreater(t4["joint_slo_goodput_compliance_pct"], 95.0)
        self.assertGreater(t4["effective_goodput_qps"], t1["effective_goodput_qps"] * 7.0)


if __name__ == "__main__":
    unittest.main()
