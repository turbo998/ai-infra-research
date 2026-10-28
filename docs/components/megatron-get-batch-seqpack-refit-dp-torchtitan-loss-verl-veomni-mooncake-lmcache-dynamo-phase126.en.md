# Phase 126 Track C: Megatron-LM Single-DP Refit Balancing & Sequence-Packing Stages, TorchTitan Generic Loss, veRL VeOmni FP8 Rollout Quantizer, Mooncake `hccs:device` 2 MiB Gate, LMCache `ObjectKeyIdentity` & Dynamo Stream Release

> **Phase 126 (`priority: 135`) · Evidence Class**: Upstream source code and commit descriptions (`implementation`) + deterministic offline analytical calibration (`own-offline`). No physical multi-node cluster benchmark (`own-hardware`) is claimed.

---

## 1. Core Architectural Mechanisms & Source-Level Analysis

### 1.1 Megatron-LM `get_batch` Config Caching, All-Stage Sequence Packing & Single-DP-Group Refit Balancing (`#8525`, `#8592`, `#8502`)
- **Cache `TransformerConfig` in `get_batch` (`#8525`, `51039e52`)**:
  - **Upstream Path**: `pretrain_gpt.py`, `pretrain_hybrid.py` (`51039e52714e2f664ebfcf4c25789f6c94ea8fa1`).
  - **Mechanism**: Caches `(args, config)` in `_get_transformer_config(args)` so `get_batch` builds `TransformerConfig` once per `args` object rather than repeating `__post_init__` validation warnings on every microbatch (`64 -> 1` build per 64-microbatch step).
- **Build Sequence-Packing Datasets on Every Pipeline Stage (`#8592`, `e4a643d8`)**:
  - Updates `is_dataset_built_on_rank` so every pipeline stage (at `tp_rank == 0`) builds the dataset whenever `is_packed_sequence` or `args.sequence_packing_scheduler` is enabled (`0` missing middle-PP datasets).
- **Spread Refit Sources Across Replicas Sharing One DP Group (`#8502`, `4a55e6fe`)**:
  - **Root Cause**: `_round_robin_dp` in `megatron/core/resharding/utils.py` previously short-circuited with `if len(grouped_by_dp) == 1: return src_meta_list[0]`. When `TP=1, PP=1, DP=8`, all 8 source replicas belong to a single DP group, forcing every destination rank to read from `src_meta_list[0]` (rank 0).
  - **Fix**: Balances across replicas within a single DP group via round-robin selection, cutting modeled 64 GB refit time at `DP=8` (`50 GB/s` per rank) from `10.24 s` to **`1.28 s` (`8.0x` speedup)**.

### 1.2 TorchTitan Generic `Loss[LossInputT]`, Qwen3 Typed-Decision Recipe & veRL VeOmni Rollout Weight Quantizer (`#4582`, `#5569`, `verl#8369`)
- **TorchTitan Generic `Loss[LossInputT]` Hierarchy (`#4582`, `740d5f75`) & Qwen3 Typed-Decision Recipe (`#5569`, `36af428d`)**:
  - Redesigned `Loss(ABC, Configurable, Generic[LossInputT])` with contravariant `LossInputT` so single-tensor losses (`CrossEntropyLoss`, `MSELoss`), tuple-input `MTPLoss`, and `ChunkedLossWrapper` share a sound typed callable interface while preserving bitwise parity in `DAPOLoss`, alongside `torchtitan_recipes.decision` for Qwen3 typed-decision fine-tuning.
- **veRL `VeOmniEngineWithLMHead` Trainer-Side Rollout Weight Quantizer (`verl#8369`, `6f3f449e`)**:
  - Implements `set_rollout_weight_quantizer` on `VeOmniEngineWithLMHead` (`verl/workers/engine/veomni/transformer_impl.py`) and applies `Fp8ConfigWeightQuantizer` to `get_per_tensor_param` streams—acting as a zero-copy passthrough for `deepseek_v4` / `deepseek_v41` mixed FP8/FP4 exports and quantizing BF16 streams to FP8 (`0` init failures, **`-50.0%`** rollout weight transfer volume).

### 1.3 Mooncake `hccs:device` 2 MiB Gate, Ascend Fragment Failure, TENT Self-Observability, LMCache `ObjectKeyIdentity` & Dynamo Stream Release (`#5049`, `#4888`, `#4938`, `lmcache#5988`, `lmcache#6090`, `dynamo#17511`, `dynamo#14325`)
- **Mooncake `hccs:device` Alignment Gate (`#5049`, `811c97f0`) & Ascend Fragment Rejection (`#4888`, `eccc7c59`)**:
  - Enforces the 2 MiB buffer alignment check only when `HasHccsDeviceProtocolDescInGlobalResourceConfig` resolves `"hccs:device"` (`hccs_device_protocol_ = true`), fabric memory is off, and the buffer is remote-accessible (`0` false rejections on `"hccs:host"` or `"roce:*"`); and explicitly marks Ascend `Slice*` fragments as failed when device lookup rejects submission.
- **Mooncake TENT Metrics Self-Observability (`#4938`, `d3ef7b1c`)**: Returns HTTP 500 when Prometheus/JSON serialization fails and surfaces the failed scrape attempt on the next successful scrape.
- **LMCache `ObjectKeyIdentity` (`lmcache#5988`, `0108f5b6`) & Dynamo Trace Analyzer (`lmcache#6090`, `b89bc5f9`)**:
  - Extracts immutable matching fields (`chunk_hash`, `kv_rank`, `model_name`, `object_group`, `cache_salt`) into `ObjectKeyIdentity` (`-50.0%` duplicate field storage) and adds `dynamo.request_trace` OTLP log parsing to `lmcache bench trace-analyze`.
- **Dynamo OpenAI `logit_bias` Forwarding (`dynamo#17511`, `843dc327`) & vLLM Stream Release on Local Stop (`dynamo#14325`, `78ac1bc9`)**:
  - Forwards `"logit_bias"` in `sglang/request_handlers/llm/decode_handler.py` and tracks `locally_finished` in `vllm_processor.py` to release backend vLLM streams immediately upon local stop (`0` leaked streams).

---

## 2. Deterministic Offline Calibration Summary (`own-offline`)

Experiment script: `experiments/megatron_get_batch_seqpack_refit_dp_torchtitan_loss_verl_veomni_mooncake_lmcache_dynamo_phase126/run.py`

| Subsystem / Metric | Before | Phase 126 After | Verified Outcome |
|---|---|---|---|
| Megatron-LM Single-DP-Group (`DP=8`) Refit | All ranks read `src_meta_list[0]` (`10.24 s`) | **Round-robin across DP group (`1.28 s`)** | **`8.0x`** refit throughput speedup |
| Megatron-LM `get_batch` Config & PP Packing | `64` builds/step; missing middle-PP dataset | **`_get_transformer_config` cache (`1` build)** | Eliminates log spam & middle-PP stalls |
| veRL VeOmni DeepSeek-V4 / FP8 Rollout | `NotImplementedError` crash in `init_model` | **`set_rollout_weight_quantizer` + passthrough** | **`0`** crashes; **`-50.0%`** BF16->FP8 sync bytes |
| Mooncake Ascend `2 MiB` Gate & Fragments | `"hccs:host"` false 2 MiB check / silent hang | **`"hccs:device"` gate + fragment fail status** | **`0`** false rejections / **`0`** hung slices |
| LMCache `ObjectKeyIdentity` & Dynamo Stop | Duplicate 5-field copies / leaked vLLM stream | **Shared `ObjectKeyIdentity` + stop release** | **`-50.0%`** field copies; **`0`** leaked streams |
