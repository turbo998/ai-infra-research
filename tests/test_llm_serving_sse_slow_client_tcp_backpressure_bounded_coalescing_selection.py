import unittest
from experiments.llm_serving_sse_slow_client_tcp_backpressure_bounded_coalescing_selection.run import evaluate_sse_slow_client_backpressure


class TestSseSlowClientBackpressure(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_sse_slow_client_backpressure()
        self.assertEqual(res["round"], 307)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_bounded_token_coalescing_ring_buffer_plus_slow_reader_ttl_eviction"]["healthy_client_p99_tpot_ms"],
            20.0,
        )
        self.assertLess(
            tiers["tier4_bounded_token_coalescing_ring_buffer_plus_slow_reader_ttl_eviction"]["gateway_memory_footprint_gb"],
            1.5,
        )


if __name__ == "__main__":
    unittest.main()
