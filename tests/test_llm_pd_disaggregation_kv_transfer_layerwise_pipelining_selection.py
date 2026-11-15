import unittest
from experiments.llm_pd_disaggregation_kv_transfer_layerwise_pipelining_selection.run import (
    evaluate_pd_disagg_layerwise_rdma_tiers,
)


class TestPdDisaggLayerwiseRdmaSelection(unittest.TestCase):
    def test_pd_layerwise_rdma_tiers(self) -> None:
        data = evaluate_pd_disagg_layerwise_rdma_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_monolithic_end_of_prefill_blind_push_stall_and_oom"]
        t4 = tiers["tier4_preallocated_slot_plus_layer_by_layer_pipelined_gpudirect_rdma"]
        self.assertGreater(t1["exposed_kv_transfer_latency_ms"], 200.0)
        self.assertLess(t4["exposed_kv_transfer_latency_ms"], 5.0)
        self.assertEqual(t4["decode_oom_or_drop_rate_pct"], 0.0)
        self.assertGreater(t4["layerwise_compute_transfer_overlap_pct"], 98.0)


if __name__ == "__main__":
    unittest.main()
