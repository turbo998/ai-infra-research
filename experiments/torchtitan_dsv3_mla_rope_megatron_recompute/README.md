# TorchTitan DeepSeek-V3 MLA Int32-Pair RoPE, FlexAttention Recompile Guard & Megatron-LM Recompute Region

Deterministic Python 3.12 standard-library model verifying:
1. `pytorch/torchtitan` commits `2fddc6b7a0d32bbe629f24e8b74977fa3b7f2643` (`#5529`) and `21db340cba613c00702ea97730cc3cb79b6d6f50` (`#5161`): `_RotateTail` rotates DeepSeek-V3 `q`'s `64` RoPE dims in-place as `32` `int32` words within the contiguous `96`-word (`192` `bf16`) row under `@local_compile("mla_qk", options={"max_pointwise_cat_inputs": 0, ...})`, halving memory operations per row (`192` vs `384`, `-50.0%`) and eliminating `100%` of masked-load `cat` ops with exact forward/backward bitwise equivalence.
2. `pytorch/torchtitan` commit `aabe6eff1e6e35379e0c3a3d5041981a1ab60d5f` (`#5442`): Raising `FlexInnerAttention` `recompile_limit` from `8` to `16` with `fullgraph=True` eliminates silent eager fallbacks on `10`-graph architectures (`12.0 ms` vs `46.6 ms`, `3.88x` speedup) while failing fast on graph breaks or a 17th graph.
3. `NVIDIA/Megatron-LM` commit `d2c430ab485c6947701447e5d551d3aafaa39161` (`#8334`): `_RecomputeRegion` (`RECORDED -> RELEASED -> REBUILT -> DONE`) + `_InputStash` eliminates autograd reference cycles (`0` vs `1`), prevents leaked Transformer Engine FP8 queue states under `no_grad` / `input_without_grad` (`0` vs `32`), and deduplicates parallel-consumer hook recomputations (`grad_enabled_per_call == [False, True]`).

## Run & Check
```bash
python experiments/torchtitan_dsv3_mla_rope_megatron_recompute/run.py --check
```
