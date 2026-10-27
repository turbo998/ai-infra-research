# Phase 113 Track C: TransformerEngine Compiled Linear+Bias & SM120 Deterministic FA4 Skip, Megatron Vision RoPE FP32 & NCCL Determinism, and Mooncake / LMCache / Dynamo / NIXL

Deterministic Python 3.12 standard-library verification suite for Phase 113 (`priority: 122`) Track C:
- `NVIDIA/TransformerEngine@45223d764a` & `6d6a500860`: `torch.compile` `BasicLinear`+`Bias` forward fusion (`2 -> 1` kernel) & SM120 (`sm_120`) deterministic backward FA4 skip (`requires_backward` + explicit `device` capability lookup)
- `NVIDIA/Megatron-LM@3ca52d95d6` & `62c063f2b7`: `Pixtral2DRotaryEmbedding` FP32 `self.freqs` retention outside `register_buffer` (`0.0` vs BF16 cos error) & `--deterministic-mode` empty `NCCL_ALGO` / `Ring,NVLS` on Blackwell SM100
- `pytorch/torchtitan@f1bb643a71`, `kvcache-ai/Mooncake@d782a5998f` & `0e626c572b`, `LMCache/LMCache@e45d4d5a85`, `ai-dynamo/dynamo@cffc4fa75c` & `ai-dynamo/nixl@7bd1d1386e`: TorchTitan RL `ActorSystem`, Mooncake DFS cached shard path (`64 -> 1` normalizations `64.0x`) & `session_ids`, LMCache `request_configs` lookup key isolation, Dynamo logprob-gated per-token decode skip (`640 -> 128` calls `5.0x` reduction), and NIXL Mooncake `postXfer` `std::vector` RAII (`128 -> 0` leaked descriptors)

```powershell
python experiments/te_compile_linear_bias_sm12x_fa4_megatron_rope32_mooncake_phase113/run.py --check
```
