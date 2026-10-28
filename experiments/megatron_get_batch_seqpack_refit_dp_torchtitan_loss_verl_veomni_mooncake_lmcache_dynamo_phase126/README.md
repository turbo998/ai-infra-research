# Phase 126 Track C: Megatron-LM Single-DP Refit Balancing & Sequence-Packing Stages, TorchTitan Generic Loss, veRL VeOmni FP8 Rollout Quantizer, Mooncake HCCS Alignment, LMCache ObjectKeyIdentity & Dynamo Stream Release

Deterministic CPU/offline analytical suite (`own-offline`) modeling:
1. **Megatron-LM `get_batch` Config Caching, All-Stage Sequence Packing & Single-DP-Group Refit Balancing (`#8525`, `#8592`, `#8502`)**: Caching `TransformerConfig` across microbatches (`64 -> 1` build/step), building packed-sequence datasets on all PP stages (`TP=0`), and round-robin balancing refit source replicas when `len(grouped_by_dp) == 1` (`8.0x` refit speedup at `DP=8`).
2. **TorchTitan Generic `Loss[LossInputT]`, Qwen3 Typed-Decision Recipe & veRL VeOmni Rollout Weight Quantizer (`#4582`, `#5569`, `verl#8369`)**: Decoupling single-tensor and tuple-based (`MTPLoss`, `ChunkedLossWrapper`) loss abstractions, and wiring `set_rollout_weight_quantizer` into `VeOmniEngineWithLMHead` (`0` init failures on DeepSeek-V4 FP8/FP4 and `-50.0%` parameter stream bytes on BF16->FP8 rollout sync).
3. **Mooncake `hccs:device` 2 MiB Gate, Ascend Fragment Failure Propagation, TENT Self-Observability, LMCache `ObjectKeyIdentity` & Dynamo Stream Release (`#5049`, `#4888`, `#4938`, `lmcache#5988`, `lmcache#6090`, `dynamo#17511`, `dynamo#14325`)**: Restricting 2 MiB alignment checks to `"hccs:device"` (`0` false rejections on `"hccs:host"` / `"roce:*"`), failing Ascend transfer slices on device lookup rejection, deduplicating `ObjectKeyIdentity` fields, forwarding OpenAI `logit_bias` to SGLang, and releasing vLLM streams immediately on local stop (`0` leaked streams).

## Run
```bash
python experiments/megatron_get_batch_seqpack_refit_dp_torchtitan_loss_verl_veomni_mooncake_lmcache_dynamo_phase126/run.py
python experiments/megatron_get_batch_seqpack_refit_dp_torchtitan_loss_verl_veomni_mooncake_lmcache_dynamo_phase126/run.py --check
```
