import unittest
from experiments.llm_serving_cudagraph_batch_bucketing_padding_waste_shared_mempool_selection.run import evaluate_cudagraph_bucketing_and_shared_pool


class TestCudaGraphBucketingAndSharedPool(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_cudagraph_bucketing_and_shared_pool()
        self.assertEqual(res["round"], 325)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_shared_mempool_plus_piecewise_cudagraph_for_mixed_chunked_prefill"]["cudagraph_reserved_hbm_gb"],
            1.2,
        )
        self.assertLess(
            tiers["tier4_shared_mempool_plus_piecewise_cudagraph_for_mixed_chunked_prefill"]["avg_batch_padding_compute_waste_pct"],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
