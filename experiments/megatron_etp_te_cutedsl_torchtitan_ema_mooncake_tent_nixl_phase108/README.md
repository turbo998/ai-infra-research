# Megatron-LM Inference Expert TP, TransformerEngine CuTeDSL Eval GEMM, TorchTitan EMA Resharding & Mooncake/NIXL Lifecycle (Phase 108)

Reproducible stdlib-only model validating:
- `NVIDIA/Megatron-LM@2abdd0d1c6` (`#4743`): Inference Expert Tensor Parallelism (`tp_ep`) across dynamic engine and token dispatchers.
- `NVIDIA/TransformerEngine@0c4d9610ac` (`#3666`): CuTeDSL single-group `DenseGEMM` enabled in `no_grad` / `frozen_weights` evaluation without columnwise quantized buffers.
- `pytorch/torchtitan@dc9ea005e4` (`#5357`): Offloaded EMA `DTensor.from_local` preserving live global `shape` and `stride` on uneven row shards.
- `kvcache-ai/Mooncake@dd4371ade4` (`#4736`), `8a4d4f68ca` (`#4904`) & `ai-dynamo/nixl@e053fc4fd4` (`#2522`): HA restore lifecycle FSM, Intel XPU `cuFile` GDS mutual exclusion, and NIXL terminal transfer/error tracing.

```bash
python experiments/megatron_etp_te_cutedsl_torchtitan_ema_mooncake_tent_nixl_phase108/run.py --check
```
