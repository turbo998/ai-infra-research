# Dynamo KV-Aware Router: Rational Overlap Credit Decay & Conditional Disaggregation Bypass

Deterministic offline simulator backed by upstream source verification of `ai-dynamo/dynamo` (`commit e3fb11d8632259cdbcea338eafa34b1e1cb04d59`):
- `lib/kv-router/src/scheduling/selector/reference.rs` (`DefaultWorkerScorer::worker_logit`)
- `lib/kv-router/src/conditional_disagg.rs` (`IslBoundingPolicy`, `DecodeLoadGate`)
- `lib/llm/src/kv_router/prefill_router/conditional_bypass.rs` (`should_bypass_remote_prefill`)

## Quick Run

```bash
python experiments/dynamo_kv_aware_router/run.py
python experiments/dynamo_kv_aware_router/run.py --check
```
