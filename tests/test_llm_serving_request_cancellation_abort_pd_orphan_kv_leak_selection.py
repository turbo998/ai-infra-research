import unittest
from experiments.llm_serving_request_cancellation_abort_pd_orphan_kv_leak_selection.run import evaluate_request_cancellation_pd_kv_leak


class TestRequestCancellationPdKvLeak(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_request_cancellation_pd_kv_leak()
        self.assertEqual(res["round"], 302)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertEqual(
            tiers["tier4_asgi_disconnect_watcher_chunk_abort_plus_nixl_lease_ttl_gc"]["decode_leaked_kv_gb_per_hour"],
            0.0,
        )
        self.assertGreater(
            tiers["tier4_asgi_disconnect_watcher_chunk_abort_plus_nixl_lease_ttl_gc"]["effective_goodput_req_per_min"],
            140.0,
        )


if __name__ == "__main__":
    unittest.main()
