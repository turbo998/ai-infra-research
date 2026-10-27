# Dynamo vLLM Sidecar Frontend Decode, TorchTitan Varlen Window Metadata & NIXL Ephemeral Port Binding (`Phase 101`)

Deterministic Python 3.12 standard-library model reproducing:
1. **Dynamo vLLM Sidecar Unconditional Frontend Decode** (`98f74e1dc8`, PR `#17324`): removes `requires_frontend_decode` conditional branching in `lib/sidecar/vllm/src/convert.rs`, always setting `output_text = Some(false)` and `output_token_ids = true` (`71.43%` per-step wire-payload reduction while preserving first-output prompt logprobs before frontend stop-string truncation).
2. **TorchTitan Unified `VarlenAttentionMetadata` Sliding Window Contract** (`b036f21aac`, PR `#5607`): models `window_size = (left, right)` in `VarlenAttentionMetadata` across packed variable-length sequences (`4K..32K`) and Ulysses Context Parallelism (`91.77%` tile reduction at `W=1024`).
3. **NIXL Initiator Ephemeral Port Binding & LMCache HF3FS Demux Teardown** (`744fc8aca1`, PR `#2456`; `d0a8680792`, PR `#6052`): models `cfg.listenPort = 0` vs fixed port `8888`, anchored `ss -Htuln` local-port regex `[:.]${next_port}$` (`0` false positives vs `3`), and HF3FS native L2 adapter demux thread shutdown.

## Usage
```bash
python experiments/dynamo_sidecar_torchtitan_varlen_nixl_phase101/run.py --check
```
