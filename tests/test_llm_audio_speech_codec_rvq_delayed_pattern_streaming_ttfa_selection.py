import unittest
from experiments.llm_audio_speech_codec_rvq_delayed_pattern_streaming_ttfa_selection.run import evaluate_realtime_speech_rvq_dual_ar


class TestRealtimeSpeechRvqDualAr(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_realtime_speech_rvq_dual_ar()
        self.assertEqual(res["round"], 299)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_hierarchical_dual_ar_7b_temporal_100m_depth_plus_causal_chunk_vocoder"]["per_frame_80ms_gen_latency_ms"],
            15.0,
        )
        self.assertGreater(
            tiers["tier4_hierarchical_dual_ar_7b_temporal_100m_depth_plus_causal_chunk_vocoder"]["audio_mos_score"],
            4.3,
        )


if __name__ == "__main__":
    unittest.main()
