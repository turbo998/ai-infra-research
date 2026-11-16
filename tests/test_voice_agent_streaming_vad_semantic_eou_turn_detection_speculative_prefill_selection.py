import unittest
from experiments.voice_agent_streaming_vad_semantic_eou_turn_detection_speculative_prefill_selection.run import evaluate_voice_agent_vad_semantic_eou


class TestVoiceAgentVadSemanticEou(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_voice_agent_vad_semantic_eou()
        self.assertEqual(res["round"], 318)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_semantic_eou_plus_speculative_prefill_on_pause_with_prefix_cache"]["turn_taking_response_latency_ms"],
            250.0,
        )
        self.assertLess(
            tiers["tier4_semantic_eou_plus_speculative_prefill_on_pause_with_prefix_cache"]["mid_sentence_false_interruption_pct"],
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
