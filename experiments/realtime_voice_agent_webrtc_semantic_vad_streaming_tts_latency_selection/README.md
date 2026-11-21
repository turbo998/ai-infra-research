# Real-Time Voice & Multimodal Agent Architecture: WebRTC, Semantic VAD, Streaming STT-LLM-TTS vs. S2S Omni Benchmark

Deterministic quantitative evaluation model for [`docs/components/realtime-voice-agent-webrtc-semantic-vad-streaming-tts-latency-selection-faq.md`](../../docs/components/realtime-voice-agent-webrtc-semantic-vad-streaming-tts-latency-selection-faq.md) and [`.en.md`](../../docs/components/realtime-voice-agent-webrtc-semantic-vad-streaming-tts-latency-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/realtime_voice_agent_webrtc_semantic_vad_streaming_tts_latency/run.py
python experiments/realtime_voice_agent_webrtc_semantic_vad_streaming_tts_latency/run.py --check
python -m unittest tests/test_realtime_voice_agent_webrtc_semantic_vad_streaming_tts_latency_selection.py
```
