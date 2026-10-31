# NCCL EFA-GDA FlushAsync/Wait, ROCm rocjitsu VMEM/GDS EXP Accounting & Mooncake/NIXL/LMCache (Phase 167)

Deterministic offline model covering:
- `NVIDIA/nccl` `d2b6b05891` & `bf40900fbd`: EFA-GDA CQ-derived per-peer `FlushAsync`/`Wait` and `_flushesAllPutsOnAnySignal` encapsulation.
- `ROCm/rocm-systems` `1340bc1f6a` (`#12583`) & `08f8ec35ef` (`#14164`): `rocjitsu` VMEM completion and CDNA4 GDS EXP race-detector accounting plus typed CLI.
- `kvcache-ai/Mooncake` `c67a88f381` (`#4875`), `ai-dynamo/nixl` `d93d752439` (`#2358`), and `LMCache/LMCache` `78b6b80f29` (`#6007`).
