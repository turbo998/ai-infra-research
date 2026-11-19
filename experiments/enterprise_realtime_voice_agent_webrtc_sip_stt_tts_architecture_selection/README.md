# Enterprise Real-Time Voice Contact Center & Multimodal Copilot Architecture Selection (`R479`)

Deterministic evaluation suite comparing four enterprise real-time voice agent and contact-center architectures across `3,000,000` monthly call minutes (`1,500` concurrent active calls, `600ms` P95 Voice-to-Voice SLA):

1. **Tier 1**: Turn-Based HTTP Audio File Upload (`Batch REST STT -> Full LLM -> Full TTS MP3`) (`4,250ms` P95 V2V latency, `1,850ms` barge-in lag, `$0.085/min`, `$255,000/mo` TCO)
2. **Tier 2**: 100% Native Speech-to-Speech Frontier Audio API (`GPT-4o Realtime Audio` on All Calls) (`380ms` P95 V2V latency, `68.0%` deterministic text guardrail auditability, `$0.240/min`, `$720,000/mo` TCO)
3. **Tier 3**: Naive WebSocket `STT -> LLM -> TTS` with Fixed 1,200ms Silence VAD (`1,680ms` P95 V2V latency, `620ms` barge-in lag, `$0.052/min`, `$156,000/mo` TCO)
4. **Tier 4**: `WebRTC/SIP` Edge (`LiveKit / ACS`) + Semantic Turn-Detector + Pipelined `Streaming STT -> Guardrailed Text LLM -> Clause TTS` (85% Calls) + `Speech-to-Speech` Escalation (15%) (`520ms` P95 V2V latency, `95ms` barge-in, `100.0%` compliance auditability, `$0.032/min`, `$96,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_realtime_voice_agent_webrtc_sip_stt_tts_architecture_selection/run.py
python experiments/enterprise_realtime_voice_agent_webrtc_sip_stt_tts_architecture_selection/run.py --check
python -m unittest tests/test_enterprise_realtime_voice_agent_webrtc_sip_stt_tts_architecture_selection.py
```
