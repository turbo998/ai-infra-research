from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestEnterpriseVoiceContactCenterRealtimeApiVsSelfHostedWebrtcTcoSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "enterprise_voice_contact_center_realtime_api_vs_self_hosted_webrtc_tco_selection"
        )

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["p95_mouth_to_ear_latency_ms"], 450.0)
        self.assertLess(data["tiers"][3]["cost_per_call_minute_usd"], 0.05)

    def test_docs_and_h5_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "enterprise-voice-contact-center-realtime-api-vs-self-hosted-webrtc-tco-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "enterprise-voice-contact-center-realtime-api-vs-self-hosted-webrtc-tco-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "enterprise-voice-contact-center-realtime-api-vs-self-hosted-webrtc-tco-selection-faq-summary.svg"
        h5 = self.repo_root / "docs" / "h5" / "voice-contact-center-realtime-api-vs-webrtc-tco-lab.html"
        sizer = self.repo_root / "docs" / "reports" / "enterprise-voice-contact-center-realtime-api-vs-self-hosted-webrtc-tco-selection-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "enterprise-voice-contact-center-realtime-api-vs-self-hosted-webrtc-tco-selection-cards.html"
        for p in (zh, en, svg, h5, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
