from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestNativeSpeechToSpeechMoshiQwenOmniVsCascadedAsrLlmTtsSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "native_speech_to_speech_moshi_qwen_omni_vs_cascaded_asr_llm_tts_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["e2e_voice_response_latency_ms"], 250.0)
        self.assertGreater(data["tiers"][3]["enterprise_rag_tool_accuracy_pct"], 96.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "native-speech-to-speech-moshi-qwen-omni-vs-cascaded-asr-llm-tts-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "native-speech-to-speech-moshi-qwen-omni-vs-cascaded-asr-llm-tts-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "native-speech-to-speech-moshi-qwen-omni-vs-cascaded-asr-llm-tts-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "native-speech-to-speech-moshi-qwen-omni-vs-cascaded-asr-llm-tts-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "native-speech-to-speech-moshi-qwen-omni-vs-cascaded-asr-llm-tts-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
