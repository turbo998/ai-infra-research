from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.realtime_voice_llm_pipeline_selection.run import RESULTS_PATH, compute_model


class TestRealtimeVoiceLlmPipelineSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_voice_pipeline_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_serial_http_cascade"]
        t2 = tiers["tier2_streaming_webrtc_cascade"]
        t3 = tiers["tier3_pure_audio_token_s2s"]
        t4 = tiers["tier4_dual_stream_inner_monologue"]

        self.assertEqual(t1["p50_v2v_latency_ms"], 2480.0)
        self.assertFalse(t1["sla_350ms_pass"])
        self.assertEqual(t2["p50_v2v_latency_ms"], 540.0)
        self.assertEqual(t2["reasoning_tool_accuracy_pct"], 100.0)
        self.assertEqual(t3["p50_v2v_latency_ms"], 195.0)
        self.assertEqual(t3["reasoning_tool_accuracy_pct"], 81.5)
        self.assertEqual(t4["p50_v2v_latency_ms"], 220.0)
        self.assertEqual(t4["p99_v2v_latency_ms"], 285.0)
        self.assertTrue(t4["sla_350ms_pass"])
        self.assertEqual(t4["reasoning_tool_accuracy_pct"], 98.4)


if __name__ == "__main__":
    unittest.main()
