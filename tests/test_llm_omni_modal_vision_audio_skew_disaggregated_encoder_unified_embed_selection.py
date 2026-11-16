import unittest
from experiments.llm_omni_modal_vision_audio_skew_disaggregated_encoder_unified_embed_selection.run import evaluate_omni_modal_disaggregated_encoders


class TestOmniModalDisaggregatedEncoders(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_omni_modal_disaggregated_encoders()
        self.assertEqual(res["round"], 309)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_disaggregated_tp1_vision_audio_encoder_pool_plus_unified_embed_rdma"]["text_request_p99_ttft_ms"],
            60.0,
        )
        self.assertGreater(
            tiers["tier4_disaggregated_tp1_vision_audio_encoder_pool_plus_unified_embed_rdma"]["cluster_throughput_req_per_sec"],
            38.0,
        )


if __name__ == "__main__":
    unittest.main()
