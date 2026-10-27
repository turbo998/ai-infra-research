# Track C Experiment (Phase 109): TransformerEngine TMA Sync & SM120 Fallback, Megatron TrainState, TorchTitan Flex Compile, Mooncake TENT & ROCm KFD VRAM, LMCache Tags, Dynamo JSON & NIXL UCX Trace

Deterministic Python 3.12 standard-library model of Phase 109 (`priority: 118`) Track C distributed training & KV transport commits:

- `NVIDIA/TransformerEngine` `39c30c577f` (`#3417`), `62b13895c2` (`#3600`)
- `NVIDIA/Megatron-LM` `7aa615b336` (`#8346`), `ebab48fc40` (`#8197`)
- `pytorch/torchtitan` `323cf2b496` (`#5547`)
- `kvcache-ai/Mooncake` `ef54c9ad46` (`#4947`), `a7f04df345` (`#4657`)
- `LMCache/LMCache` `7e4f499a20` (`#5919`), `ai-dynamo/dynamo` `b79946c82f` (`#16923`), `ai-dynamo/nixl` `86d9626e13` (`#2403`)

## Run

```bash
python experiments/te_tma_sync_sm120_megatron_torchtitan_mooncake_kfd_nixl_phase109/run.py
python experiments/te_tma_sync_sm120_megatron_torchtitan_mooncake_kfd_nixl_phase109/run.py --check
```
