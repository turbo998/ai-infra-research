# Enterprise Legacy Contact Center (`DTMF IVR / SIP PBX`) to AI Voice Agent Migration Selection (`R500`)

Deterministic quantitative evaluation model comparing four enterprise contact-center IVR modernization and migration architectures across **5,000,000 monthly inbound voice calls**:
- **Tier 1**: Legacy 5-Level `DTMF` Touch-Tone IVR Tree ("Press 1 for X, Press 2 for Y")
- **Tier 2**: "Big-Bang" 100% Rip-and-Replace with Pure End-to-End Voice LLM (Zero DTMF Fallback)
- **Tier 3**: VXML-to-HTTP Webhook Bridge (Utterance-Blocking STT -> LLM -> TTS Over Legacy PBX)
- **Tier 4**: `SIP REC / RTP Media Forking` Shadow Validation -> Intent-Gated Canary (`10%->50%->85%`) + Hybrid `RFC 4733 DTMF` Secure Vault + `580ms` Streaming Voice Agent

## Verification

```bash
python experiments/enterprise_multi_modal_contact_center_ivr_modernization_migration_selection/run.py --check
python -m unittest tests/test_enterprise_multi_modal_contact_center_ivr_modernization_migration_selection.py
```
