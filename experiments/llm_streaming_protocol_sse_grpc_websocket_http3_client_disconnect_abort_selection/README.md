# LLM Streaming Transport & Connection Governance Experiment (`SSE` vs. `gRPC` vs. `WebSockets` vs. `HTTP/3 QUIC` & Abort Propagation)

Reproducible analytical model comparing 4 streaming transport and connection-cancellation architectures across `10,000` concurrent LLM streams (`25%` early user stop/cancel rate, `2%` mobile packet loss, `64x H100` cluster):
- **Tier 1 (`HTTP/1.1 SSE` with `proxy_buffering on` & no disconnect cancel)**: `780 ms` client P99 TTFT (`+420 ms` proxy buffer penalty), `21.25%` zombie GPU decode token waste (`$43,200/month` wasted).
- **Tier 2 (`Tuned HTTP/2 SSE` with `X-Accel-Buffering: no` + TCP/HTTP2 abort propagation)**: `368 ms` P99 TTFT, `0.35%` zombie GPU waste.
- **Tier 3 (`Internal gRPC Bidirectional Protobuf Streaming` + native `context.Cancel`)**: `352 ms` P99 TTFT, `0.15%` zombie GPU waste.
- **Tier 4 (`Edge HTTP/3 QUIC SSE / WebTransport` + `Internal gRPC` + `Instant vLLM abort_request`, Recommended)**: `345 ms` P99 TTFT, `42 ms` mobile P99 ITL jitter under `2%` packet loss (`8.1x` smoother than TCP), and `0.12%` zombie waste (`-$42,960/month` saved).

## Run & Verify

```bash
python experiments/llm_streaming_protocol_sse_grpc_websocket_http3_client_disconnect_abort_selection/run.py
python experiments/llm_streaming_protocol_sse_grpc_websocket_http3_client_disconnect_abort_selection/run.py --check
```
