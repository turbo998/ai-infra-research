# Enterprise Real-Time Voice AI Contact Center (`<=300ms` Mouth-to-Ear) Architecture & Sizer (`#513`)

Reproducible analytical model comparing **Blocking Serial Cascade (`STT -> LLM -> TTS`)**, **Commercial Native Speech-to-Speech API (`GPT-4o Realtime / Gemini Live`)**, and **Streaming Speculative Clause-Chunked Cascade (`LiveKit WebRTC + Streaming Conformer STT + SGLang Speculative Prefill + Streaming TTS`)** across `1,000` concurrent enterprise contact center lines.

## Run

```bash
python experiments/enterprise_voice_ai_contact_center_cascaded_vs_native_s2s_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-voice-ai-contact-center-cascaded-vs-native-s2s-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-voice-ai-contact-center-cascaded-vs-native-s2s-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/voice-ai-contact-center-cascaded-vs-native-s2s-sizer-lab.html](../../docs/h5/voice-ai-contact-center-cascaded-vs-native-s2s-sizer-lab.html)
