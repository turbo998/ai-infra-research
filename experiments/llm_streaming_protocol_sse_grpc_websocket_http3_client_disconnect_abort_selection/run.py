from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "llm_streaming_protocol_sse_grpc_websocket_http3_client_disconnect_abort_selection",
        "title": "LLM Streaming Transport & Connection Governance: SSE vs. gRPC vs. WebSockets vs. HTTP/3 QUIC & Client Disconnect Abort Propagation",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "concurrent_streams": 10000,
            "early_user_stop_rate_pct": 25.0,
            "avg_abort_position_pct": 15.0,
            "mobile_packet_loss_pct": 2.0,
            "cluster_gpus": 64,
        },
        "tiers": [
            {
                "tier": "Tier 1: Unbuffered-Unaware HTTP/1.1 SSE (proxy_buffering on, No Disconnect Cancel)",
                "client_p99_ttft_ms": 780.0,
                "mobile_2pct_loss_p99_itl_jitter_ms": 340.0,
                "zombie_decode_gpu_waste_pct": 21.25,
                "monthly_wasted_gpu_cost_usd": 43200,
                "recommended_scenario": "Anti-pattern (4KB proxy buffer stalls chunks; dead sockets burn 21% GPU tokens)",
            },
            {
                "tier": "Tier 2: Tuned HTTP/2 SSE (X-Accel-Buffering: no) + TCP FIN/RST Abort Propagation",
                "client_p99_ttft_ms": 368.0,
                "mobile_2pct_loss_p99_itl_jitter_ms": 285.0,
                "zombie_decode_gpu_waste_pct": 0.35,
                "monthly_wasted_gpu_cost_usd": 710,
                "recommended_scenario": "Standard web/desktop Chat & Copilot text streaming over reliable networks",
            },
            {
                "tier": "Tier 3: East-West gRPC Bidirectional Protobuf Streaming + Native Context Cancellation",
                "client_p99_ttft_ms": 352.0,
                "mobile_2pct_loss_p99_itl_jitter_ms": 260.0,
                "zombie_decode_gpu_waste_pct": 0.15,
                "monthly_wasted_gpu_cost_usd": 305,
                "recommended_scenario": "Internal Gateway-to-Router-to-vLLM/Dynamo/Triton microservice mesh",
            },
            {
                "tier": "Tier 4 (Recommended): Edge HTTP/3 QUIC SSE / WebTransport + Internal gRPC + Instant vLLM Abort",
                "client_p99_ttft_ms": 345.0,
                "mobile_2pct_loss_p99_itl_jitter_ms": 42.0,
                "zombie_decode_gpu_waste_pct": 0.12,
                "monthly_wasted_gpu_cost_usd": 240,
                "recommended_scenario": "Global mobile/edge + enterprise AI Gateway + real-time voice/multimodal streaming",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
