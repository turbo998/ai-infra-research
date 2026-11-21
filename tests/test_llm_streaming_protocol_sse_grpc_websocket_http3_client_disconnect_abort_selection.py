from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestLlmStreamingProtocolSseGrpcWebsocketHttp3ClientDisconnectAbortSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = self.repo_root / "experiments" / "llm_streaming_protocol_sse_grpc_websocket_http3_client_disconnect_abort_selection"

    def test_run_and_check(self) -> None:
        subprocess.run([sys.executable, str(self.exp_dir / "run.py")], check=True)
        subprocess.run([sys.executable, str(self.exp_dir / "run.py"), "--check"], check=True)
        data = json.loads((self.exp_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["tiers"]), 4)
        self.assertLess(data["tiers"][3]["zombie_decode_gpu_waste_pct"], 0.5)
        self.assertLess(data["tiers"][3]["mobile_2pct_loss_p99_itl_jitter_ms"], 50.0)

    def test_docs_and_assets_exist(self) -> None:
        zh = self.repo_root / "docs" / "components" / "llm-streaming-protocol-sse-grpc-websocket-http3-client-disconnect-abort-selection-faq.md"
        en = self.repo_root / "docs" / "components" / "llm-streaming-protocol-sse-grpc-websocket-http3-client-disconnect-abort-selection-faq.en.md"
        svg = self.repo_root / "assets" / "svg" / "llm-streaming-protocol-sse-grpc-websocket-http3-client-disconnect-abort-selection-faq-summary.svg"
        sizer = self.repo_root / "docs" / "reports" / "llm-streaming-protocol-sse-grpc-websocket-http3-client-disconnect-abort-sizer.html"
        cards = self.repo_root / "docs" / "reports" / "llm-streaming-protocol-sse-grpc-websocket-http3-client-disconnect-abort-selection-cards.html"
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing required deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
