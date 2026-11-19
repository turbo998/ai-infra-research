from __future__ import annotations

import unittest
from experiments.enterprise_realtime_voice_agent_webrtc_sip_stt_tts_architecture_selection.run import build_payload


class TestEnterpriseRealtimeVoiceAgentWebrtcSipSttTtsArchitectureSelection(unittest.TestCase):
    def test_tier4_sub_600ms_latency_full_compliance_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLess(tier4["voice_to_voice_p95_response_latency_ms"], 550.0)
        self.assertLess(
            tier4["voice_to_voice_p95_response_latency_ms"],
            tier3["voice_to_voice_p95_response_latency_ms"] * 0.35,
        )
        self.assertLess(tier4["barge_in_interruption_reaction_ms"], 100.0)
        self.assertEqual(tier4["compliance_transcript_and_dlp_auditability_pct"], 100.0)
        self.assertLess(
            tier4["monthly_3m_minute_contact_center_tco_usd"],
            tier2["monthly_3m_minute_contact_center_tco_usd"] * 0.15,
        )
        self.assertLess(
            tier4["monthly_3m_minute_contact_center_tco_usd"],
            tier1["monthly_3m_minute_contact_center_tco_usd"] * 0.40,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
