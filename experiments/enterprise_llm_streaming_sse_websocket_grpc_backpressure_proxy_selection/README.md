# Enterprise LLM Streaming Protocol (`SSE vs. WebSocket vs. gRPC`), Proxy Unbuffering & Client Disconnect Cancellation (`R507`)

Deterministic quantitative evaluation model comparing four enterprise LLM streaming and reverse-proxy architectures across **50B monthly streamed tokens (`28%` user early-stop / tab-close rate)**:
- **Tier 1**: Default Reverse-Proxy Buffered HTTP (`proxy_buffering on` + `60s` Idle Timeout + Zero Abort Propagation)
- **Tier 2**: Stateful WebSockets Across Multi-Hop L7 Corporate Firewalls Without Stream Offset Resumption
- **Tier 3**: Unbuffered HTTP/2 `SSE` Without Upstream `vLLM / Cloud API` Cancellation Propagation or `Last-Event-ID` Buffer
- **Tier 4**: Edge HTTP/2 `SSE` (`X-Accel-Buffering: no` + `15s` Heartbeat) + Internal `gRPC` + End-to-End `AbortSignal -> vLLM Engine.abort()` + `Redis Stream Last-Event-ID` Resume

## Verification

```bash
python experiments/enterprise_llm_streaming_sse_websocket_grpc_backpressure_proxy_selection/run.py --check
python -m unittest tests/test_enterprise_llm_streaming_sse_websocket_grpc_backpressure_proxy_selection.py
```
