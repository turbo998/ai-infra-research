import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestRealtimeVoiceAgentWebrtcSemanticVadSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "realtime_voice_agent_webrtc_semantic_vad_streaming_tts_latency_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_script_check_and_invariants(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"], "synthetic + anonymized_reported")
        self.assertFalse(data["independently_reproduced"])
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_tcp_websocket_batch_stt_full_sentence_llm_and_batch_tts"]
        t4 = tiers["tier4_webrtc_semantic_turn_detector_speculative_clause_tts_and_pd_llm"]
        self.assertLess(t4["p95_mouth_to_ear_latency_ms"], 450.0)
        self.assertLess(t4["cost_per_voice_minute_usd"], t1["cost_per_voice_minute_usd"] * 0.25)

    def test_artifacts_exist(self) -> None:
        slug = "realtime-voice-agent-webrtc-semantic-vad-streaming-tts-latency"
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.md").exists())
        self.assertTrue((self.repo_root / "docs" / "components" / f"{slug}-selection-faq.en.md").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-sizer.html").exists())
        self.assertTrue((self.repo_root / "docs" / "reports" / f"{slug}-selection-cards.html").exists())
        self.assertTrue((self.repo_root / "assets" / "svg" / f"{slug}-selection-faq-summary.svg").exists())


if __name__ == "__main__":
    unittest.main()
