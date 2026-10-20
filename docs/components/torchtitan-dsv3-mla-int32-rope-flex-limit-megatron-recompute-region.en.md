# TorchTitan DeepSeek-V3 MLA In-Row `int32` RoPE Rotation, FlexAttention Recompile Guard & Megatron-LM `CheckpointWithoutOutput` `_RecomputeRegion` State Machine

> **中文版**: [torchtitan-dsv3-mla-int32-rope-flex-limit-megatron-recompute-region.md](torchtitan-dsv3-mla-int32-rope-flex-limit-megatron-recompute-region.md)

- **Queue Question ID**: `torchtitan-dsv3-mla-int32-rope-flex-limit-megatron-recompute-region` (`分布式训练`, `priority: 39`, `advanced`)
- **Evidence Class**: `implementation` (verified first-hand upstream commits) + `own-offline` (deterministic Python 3.12 standard-library verification)
- **Verified Date**: 2026-10-21 (UTC+08:00)
- **Primary Upstream Sources**:
  - `pytorch/torchtitan` commit [`2fddc6b7a0d32bbe629f24e8b74977fa3b7f2643`](https://github.com/pytorch/torchtitan/commit/2fddc6b7a0d32bbe629f24e8b74977fa3b7f2643) (PR `#5529`, 2026-10-20): `torchtitan/models/common/rope.py` (blob `639efaa597`), `torchtitan/models/deepseek_v3/attention.py` (blob `f0c2f4d24e`)
  - `pytorch/torchtitan` commit [`21db340cba613c00702ea97730cc3cb79b6d6f50`](https://github.com/pytorch/torchtitan/commit/21db340cba613c00702ea97730cc3cb79b6d6f50) (PR `#5161`, 2026-10-20): `torchtitan/models/deepseek_v3/attention.py` (blob `dc7da03e85`)
  - `pytorch/torchtitan` commit [`aabe6eff1e6e35379e0c3a3d5041981a1ab60d5f`](https://github.com/pytorch/torchtitan/commit/aabe6eff1e6e35379e0c3a3d5041981a1ab60d5f) (PR `#5442`, 2026-10-20): `torchtitan/models/common/attention/attention.py` (blob `ad23ce5b43`), `torchtitan/distributed/utils.py` (blob `9de899fd23`)
  - `NVIDIA/Megatron-LM` commit [`d2c430ab485c6947701447e5d551d3aafaa39161`](https://github.com/NVIDIA/Megatron-LM/commit/d2c430ab485c6947701447e5d551d3aafaa39161) (PR `#8334`, 2026-10-20): `megatron/core/tensor_parallel/random.py` (blob `578e2f5244`)

---

## 1. Architectural Problem Statement

1. **DeepSeek-V3 MLA Strided `q_pe` Slicing & Inductor Masked-Load Concatenation (`#5529`, `#5161`)**: Splitting `q` (`[T, H, 128 + 64]`) before RoPE produces a strided `q_pe` view that prevents `int32` pair loads and causes Inductor's pointwise `cat` to emit slow masked loads.
2. **Silent Eager Fallback in `FlexInnerAttention` at Default `recompile_limit=8` (`#5442`)**: Models like Muse Spark require `10` distinct FlexAttention graphs across layers; exceeding the default limit of `8` silently drops the 9th and 10th graphs to eager execution.
3. **Megatron-LM `CheckpointWithoutOutput` Reference Cycles & Leaked TE FP8 State (`#8334`)**: Legacy `CheckpointWithoutOutputFunction` closed a reference cycle through `outputs`, pushed unconsumed states into Transformer Engine's `_is_first_fp8_module` queue when called under `no_grad` or without grad-requiring inputs, and lacked an explicit region state machine for multi-hook parallel consumers.

---

## 2. First-Hand Code Verification & Deterministic Results (`--check`)

- **In-Row `int32` `_RotateTail` (`2fddc6b7a0`, `21db340cba`)**: Views contiguous `bf16` rows as `96` `int32` words, splits `64` kept words and `32` tail words, unpacks `(pairs << 16).view(fp32)` / `(pairs & ~0xFFFF).view(fp32)`, and concatenates with `max_pointwise_cat_inputs: 0`, cutting memory ops per row from `384` to **`192` (`-50.0%`)** with `0` masked loads and bitwise-identical forward/backward outputs.
- **FlexAttention `recompile_limit=16, fullgraph=True` (`aabe6eff1e`)**: Compiles all `10/10` Muse Spark graphs (`12.0 ms` vs `46.6 ms`, **`3.88x` speedup**, `0` eager fallbacks) and fails fast on graph breaks or a 17th graph.
- **`_RecomputeRegion` State Machine (`d2c430ab48`)**: Transitions monotonically through `RECORDED -> RELEASED -> REBUILT -> DONE`, eliminating reference cycles (`0` vs `1`), preventing `32/32` leaked TE FP8 queue entries on eval/no-grad calls, and executing `rebuild()` exactly once across parallel hook consumers (`grad_enabled_per_call == [False, True]`).

---

## 3. Linked Evidence IDs

- `claim-torchtitan-dsv3-mla-int32-rope-tail-flex-limit`
- `claim-megatron-checkpoint-without-output-recompute-region`
- `claim-torchtitan-mla-rope-megatron-recompute-offline`
