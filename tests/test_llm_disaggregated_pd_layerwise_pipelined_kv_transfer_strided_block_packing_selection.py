import unittest
from experiments.llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing_selection.run import (
    evaluate_pd_layerwise_kv_transfer_tiers,
)


class TestPdLayerwiseKvTransferSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_pd_layerwise_kv_transfer_tiers()
        self.assertAlmostEqual(res["naive_post_prefill_stall_ms"], 148.0, places=1)
        self.assertAlmostEqual(res["tier4_post_prefill_stall_ms"], 4.2, places=1)
        self.assertAlmostEqual(res["stall_reduction_x"], 35.2, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["rdma_wqes_per_request"], 16000)
        self.assertEqual(t4["rdma_wqes_per_request"], 80)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
