# Enterprise Real-Time Full-Duplex Voice Agent (`WebRTC + Streaming ASR/LLM/TTS` vs. `Native Omni Speech-to-Speech`) Latency & GPU Sizing (`#466`)

Reproducible analytical model comparing Turn-Based REST Cascades, WebSocket Streaming Cascades, WebRTC Full-Duplex Heterogeneous (`L40S ASR/TTS + H200 LLM`) Pools, and Native `12.5 Hz` Omni Audio-Token Engines against the `< 250ms` human conversational Time-To-First-Audio (`TTFA`) SLA.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_realtime_voice_agent_webrtc_asr_llm_tts_omni_latency_sizing_selection/run.py --check`
