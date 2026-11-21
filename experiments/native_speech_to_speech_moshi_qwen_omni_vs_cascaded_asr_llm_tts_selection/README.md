# Native Speech-to-Speech (`Moshi / Qwen2.5-Omni`) vs. Cascaded `ASR + LLM + TTS` Selection Experiment

Reproducible analytical model comparing 4 real-time voice model paradigms across `200` concurrent voice sessions on `NVIDIA H100 80GB`:
- **Tier 1 (`Turn-Based Batch Whisper -> LLM -> Sentence TTS`)**: `2,150 ms` E2E voice latency, `96.5%` tool accuracy, `5.0%` emotion retention.
- **Tier 2 (`Streaming Clause-Pipelined ASR + vLLM + CosyVoice2`)**: `480 ms` E2E voice latency, `96.8%` tool accuracy, `42` sessions/H100.
- **Tier 3 (`Pure Native Speech-to-Speech Moshi / GLM-4-Voice` with `12.5Hz RVQ`)**: `195 ms` E2E voice latency, `95.0%` emotion retention, but `78.5%` complex tool accuracy.
- **Tier 4 (`Hybrid Thinker-Talker Omni Architecture Qwen2.5-Omni`, Recommended)**: `235 ms` E2E voice latency, `96.4%` tool accuracy, `92.0%` emotion retention, and `48` concurrent sessions/H100.

## Run & Verify

```bash
python experiments/native_speech_to_speech_moshi_qwen_omni_vs_cascaded_asr_llm_tts_selection/run.py
python experiments/native_speech_to_speech_moshi_qwen_omni_vs_cascaded_asr_llm_tts_selection/run.py --check
```
