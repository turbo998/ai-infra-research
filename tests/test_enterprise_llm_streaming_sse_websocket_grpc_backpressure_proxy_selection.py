from __future__ import annotations

import unittest
from experiments.enterprise_llm_streaming_sse_websocket_grpc_backpressure_proxy_selection.run import build_payload


class TestEnterpriseLlmStreamingSseWebsocketGrpcBackpressureProxySelection(unittest.TestCase):
    def test_tier4_eliminates_proxy_buffering_lag_zombie_tokens_and_midstream_drops(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLess(tier4["perceived_client_ttft_p99_ms"], 400.0)
        self.assertLess(
            tier4["perceived_client_ttft_p99_ms"],
            tier1["perceived_client_ttft_p99_ms"] * 0.05,
        )
        self.assertEqual(tier4["mid_stream_timeout_drop_rate_pct"], 0.0)
        self.assertLess(
            tier4["wasted_zombie_decode_tokens_on_abort_pct"],
            tier3["wasted_zombie_decode_tokens_on_abort_pct"] * 0.05,
        )
        self.assertGreater(
            tier4["mobile_network_reconnect_resume_rate_pct"],
            tier2["mobile_network_reconnect_resume_rate_pct"] + 80.0,
        )
        self.assertLess(
            tier4["monthly_50b_token_streaming_waste_and_ops_tco_usd"],
            tier1["monthly_50b_token_streaming_waste_and_ops_tco_usd"] * 0.18,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
