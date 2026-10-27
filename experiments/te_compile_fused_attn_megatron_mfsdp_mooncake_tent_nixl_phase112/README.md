# Phase 112 Track C: TE Compiled FusedAttention & Deterministic dGLU, Megatron MFSDP Readiness, TorchTitan EMA Reseed, Mooncake/Dynamo/NIXL

Deterministic stdlib model covering 10 first-hand commits across `NVIDIA/TransformerEngine`, `NVIDIA/Megatron-LM`, `pytorch/torchtitan`, `kvcache-ai/Mooncake`, `LMCache/LMCache`, `ai-dynamo/dynamo`, and `ai-dynamo/nixl`:
- `TransformerEngine@7e7898fa82` & `TransformerEngine@c25fa27525`: `torch.compile(fullgraph=True)` custom op for cuDNN `FusedAttention` with native packed QKV gradients & deterministic `dprob` feature-detection for `dGLU`.
- `Megatron-LM@17fc7ba109` & `Megatron-LM@74c9420e35`: MFSDP per-`FsdpParameterGroup` `_unshard_events` readiness (`1` BF16 event vs `2` MXFP8 events) & DTensor `_clear_sharding_prop_cache()` teardown.
- `torchtitan@ba53205255`: `BaseCheckpointManager.load` unified EMA reseed after model-only or EMA-excluded loads across all backends.
- `Mooncake@edd323921d` & `Mooncake@e3e9d70eea`: `mooncake-store-rs` hard control-plane tenant binding & TENT `CudaPlatform::copy()` exact `CUcontext` restoration.
- `LMCache@e0e92b8356`, `dynamo@492108cd43`, `nixl@f7533eb8a5`: `PeriodicThread` live worker overlap guard, HTTP/1.x `Connection: close` on drain `503`, and POSIX partial enqueue rollback via `enqueued_ios_`.
