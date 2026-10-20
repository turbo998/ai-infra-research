# TorchTitan SimpleFSDP Inspection Guard, Kimi-K3 MXFP4 Adapter & Megatron-LM Idle Rerun Strict Checkpoint Loading

Deterministic Python 3.12 standard-library model (`own-offline`) verifying:
1. `pytorch/torchtitan@3b1714ab` (`#5560`): `with disable_active_parametrization():` around `linear_param_shard_placements(model)` in `simple_fsdp.data_parallel()` avoiding `meta`-device `ReplicateComputation` materialization.
2. `pytorch/torchtitan@7d563223` (`#5337`): `KimiK3StateDictAdapter` MXFP4 `weight_packed` remapping (`249,924` HF keys across `96` safetensors files), text-only vision filtering, and KDA `A_log`/`dt_bias` DTensor layout preservation.
3. `NVIDIA/Megatron-LM@e564873e` (`#7222`): Strict (`raise_all`) checkpoint loading with `RerunState.NOT_RUNNING_YET` shards without resuming a bogus rerun.
