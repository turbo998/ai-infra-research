# TorchTitan SimpleFSDP Inspection Guard, Kimi-K3 MXFP4 State-Dict Adapter & Megatron-LM Idle Rerun Strict Checkpoint Loading

> **中文版**: [torchtitan-fsdp-inspection-guard-kimi-k3-mxfp4-adapter-megatron-idle-rerun.md](torchtitan-fsdp-inspection-guard-kimi-k3-mxfp4-adapter-megatron-idle-rerun.md)  
> **Evidence Class**: `implementation` + `own-offline`  
> **Experiment**: [`experiments/torchtitan_fsdp_inspect_kimi_k3_mxfp4_megatron_rerun/run.py`](../../experiments/torchtitan_fsdp_inspect_kimi_k3_mxfp4_megatron_rerun/run.py)

---

## 1. Engineering Problem & Context

Large-scale distributed training (FSDP + EP + PP) and quantized checkpoint conversion require strict separation between metadata inspection, tensor coordinate transformations, and state-machine resumption:
1. **TorchTitan GraphTrainer `meta`-Device FSDP Inspection Crash (`pytorch/torchtitan@3b1714ab`, `#5560`)**: In `graph_trainer_dist_moe_mxfp8_fsdp_ep_pp`, calling `simple_fsdp.data_parallel()` on a module already parametrized by GraphPP ran `linear_param_shard_placements(model)` with active parametrizations enabled. Reading `child.weight` invoked `ReplicateComputation` and tried to materialize MXFP8 compute operands while the model was still on the `meta` device (`weight.is_cuda == False`), failing in `dist_moe` with `ValueError: weight must be contiguous CUDA storage`.
2. **TorchTitan Kimi-K3 `compressed-tensors` MXFP4 Checkpoint Adapter (`pytorch/torchtitan@7d563223`, `#5337`)**: The released Kimi-K3 checkpoint spans `96` safetensors files (`249,924` HF keys), stores routed experts as `weight_packed` / `weight_scale`, pads KDA `A_log` from `n_heads` to `head_dim`, and flattens `dt_bias`.
3. **Megatron-LM Strict Checkpoint Loading with Idle Rerun State (`NVIDIA/Megatron-LM@e564873e`, `#7222`)**: Idle checkpoints save a `rerun_state_machine` shard (`RerunState.NOT_RUNNING_YET`) to preserve cached save plans, but `validate_state_dict` returned `False` on load, causing `dist_ckpt_strictness="raise_all"` to fail on the unrequested shard.

---

## 2. First-Hand Code Analysis & Quantitative Results

### 2.1 TorchTitan `disable_active_parametrization()` Guard
In `torchtitan/experiments/graph_trainer/simple_fsdp.py` (`3b1714ab`), wrapping `linear_param_shard_placements(model)` inside `with disable_active_parametrization():` ensures FSDP inspection reads raw `_ShardedFSDPTensor` shards without executing `ReplicateComputation` or `_build_operands` (`0` operand builds on `meta`).

### 2.2 Kimi-K3 MXFP4 `QuantizedHuggingFaceStorageReader` and KDA DTensor Redistribution
In `torchtitan/models/kimi_k3/state_dict_adapter.py` (`7d563223`):
- **Index Remapping & Vision Filtering**: `get_hf_storage_reader(..., from_quantized=True)` uses `QuantizedHuggingFaceStorageReader(thread_count=4, target_dtype=torch.bfloat16)`. `fqn_to_index_mapping` remaps `X.weight_packed -> X.weight` and drops `X.weight_scale` across all `249,924` HF keys in `96` files, while filtering `vision_tower.*` and `mm_projector.*` when `vision_encoder is None` so DCP consolidation never writes empty dtype headers.
- **KDA `A_log` and `dt_bias` DTensor Coordinate Safety**: Before padding `A_log` by `head_dim - n_heads` (`128 - 16 = 112` elements when `to_quantized_layout=True`) or flattening `dt_bias` (`[16, 128] -> [2048]`), `to_hf()` records `(device_mesh, placements)` in `_kda_parameter_layouts` and redistributes DTensors to `(Replicate(),) * ndim`; `from_hf()` slices `A_log[:n_heads]` / reshapes `dt_bias` to `[n_heads, head_dim]` and restores the original mesh placements.

### 2.3 Megatron-LM Idle Rerun Schema Inclusion Without Rerun Activation
In `megatron/training/checkpointing.py` (`e564873e`), checking `if restore_rerun_state or idle_rerun_state:` populates `gen_sd_rerun_state` so strict load (`raise_all`) accepts the `NOT_RUNNING_YET` shard (`0` unexpected keys), while setting `ignore_rerun_state = not restore_rerun_state` (`True` when idle) prevents resuming a bogus rerun (`rerun_requested == False`).

---

## 3. Architect Exercises & Falsifiable Questions

1. **Design Exercise**: Why must `A_log` and `dt_bias` DTensors be redistributed to `Replicate()` before padding or flattening in `KimiK3StateDictAdapter.to_hf()`?
   - **Answer**: Padding along the last dimension or flattening `[n_heads, head_dim] -> [n_heads * head_dim]` alters global tensor coordinates across shard boundaries; replicating first guarantees valid local tensor views before restoring the sharded placement in `from_hf()`.
2. **Falsifiable Hypothesis**: Does calling `.contiguous()` on the weight tensor before `dist_moe` resolve the B200 GraphTrainer failure? No—CI run `39374680382` proved the failure was triggered because `weight.is_cuda` was `False` on the `meta` device during parameter inspection.
