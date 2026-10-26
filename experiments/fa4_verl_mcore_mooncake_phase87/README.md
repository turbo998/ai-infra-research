# FA4 Paged-KV CUDA Graph & SM100 HD256 SplitKV, TorchTitan GB300 4D RL Topology, Mooncake NoF & LMCache Packed Token IPC (Phase 87)

Deterministic Python 3.12 stdlib model auditing:
- `Dao-AILab/flash-attention#2984` (`73019affed`) & `#2917` (`e9cf2c1651`): Skipping `.max().item()` host sync during paged-KV CUDA graph capture (`CaptureStatus::None` guard) and SM100 `hd256` 2CTA forward `SplitKV` work-tile predicate (`has_kv_work`).
- `pytorch/torchtitan#5581` (`43d1bb9801`): Muse Spark 816B-A42B DAPO-Math on GB300 (`FSDP 8 x TP 4 x CP 4 x EP 16` on `128` GPUs vs `FSDP 64 x TP 4 x EP 4` on `256` GPUs; `8x` smaller expert FSDP group, `15% -> 28%` MFU).
- `volcengine/verl#8284` (`d430157d15`), `NVIDIA/Megatron-LM#6502` (`2052769557`) & `#7926` (`7d3f0fb3b5`).
- `kvcache-ai/Mooncake#3424` (`0afc417657`) & `LMCache/LMCache#5150` (`6913259264`) / `#4410` (`00f6d71ed1`): SPDK-decoupled `NVMeoFInitiator` / `DmaBufferAllocator` with refcounted `SpdkEnvGuard` and packed `uint32` token buffers (`9.0x` memory footprint reduction vs Python int tuples).

## Run
```bash
python experiments/fa4_verl_mcore_mooncake_phase87/run.py
python experiments/fa4_verl_mcore_mooncake_phase87/run.py --check
```
