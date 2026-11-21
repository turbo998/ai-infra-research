# Experiment: Enterprise AI Voice Contact Center Architecture Selection — Realtime S2S API vs. Streaming WebRTC + SGLang (`#409`)

Reproducible analytical model comparing `Legacy HTTP Turn-by-Turn`, `Commercial Native Speech-to-Speech (GPT-4o Realtime API)`, and `Streaming WebRTC + Semantic VAD + Fast-ASR + SGLang + 6-Word Clause TTS` across Mouth-to-Ear (`M2E`) latency, pre-synthesis compliance guardrails, and cost per call-minute at `10M minutes/month`.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_voice_contact_center_realtime_api_vs_self_hosted_webrtc_tco_selection/run.py`
- **Verify**: `python experiments/enterprise_voice_contact_center_realtime_api_vs_self_hosted_webrtc_tco_selection/run.py --check`
