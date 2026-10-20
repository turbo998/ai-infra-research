# TorchTitan Kimi K3 MXFP4 Weight Adapter (`KimiK3StateDictAdapter`), DeepSeek-V3 HF Save Index Alignment, and GraphTrainer `SimpleFSDP` Zero-Materialization Meta Inspection

> **中文版**: [torchtitan-kimi-k3-mxfp4-adapter-graph-fsdp-meta-inspect.md](torchtitan-kimi-k3-mxfp4-adapter-graph-fsdp-meta-inspect.md) | **Associated Experiment**: [`experiments/torchtitan_kimi_k3_mxfp4_adapter_fsdp/README.md`](../../experiments/torchtitan_kimi_k3_mxfp4_adapter_fsdp/README.md)

### One-Sentence Takeaway

PyTorch TorchTitan landed three critical checkpointing and graph-parallel contracts for hybrid-attention and large-scale MoE models: (1) **`KimiK3StateDictAdapter` (`#5337`)** loads `compressed-tensors` MXFP4 routed experts (`.weight_packed` / `.weight_scale`) into BF16 via `QuantizedHuggingFaceStorageReader(target_dtype=torch.bfloat16)` while redistributing KDA (`KimiDeltaAttention`) `A_log` and `dt_bias` `DTensor`s to `Replicate()` before padding `A_log` (`[n_heads] -> [head_dim]`) and flattening `dt_bias` (`[n_heads, head_dim] -> [n_heads * head_dim]`), restoring their exact `DTensor` mesh and placements in `from_hf()`; (2) **`DeepSeekV3StateDictAdapter` (`#5065`)** filters `45,808` unsaved FP8 `.weight_scale_inv` entries and `788` unused MTP layer-61 entries out of the released `91,991`-entry `model.safetensors.index.json` (matching the exact `45,395` / `46,183` written tensors so DCP consolidation never emits `{"shape": [], "dtype": ""}` headers that crash `safetensors` with `unknown variant ''`), and clones independent storage for `_MTP_SHARED_COPIES` (`+3.71 GB` on 671B BF16); and (3) **`GraphTrainer` `SimpleFSDP` (`#5560`)** wraps `linear_param_shard_placements(model)` in `with disable_active_parametrization():` so re-applying `data_parallel()` after `GraphPP + EP` does not invoke `ReplicateComputation` or MXFP8 `_build_operands()` while parameters reside on the `meta` device.

### Component Responsibilities and Upstream Locations

| Module & File | Upstream Repository & Commit / Blob SHA | Core Engineering Responsibility |
| :--- | :--- | :--- |
| `torchtitan/models/kimi_k3/state_dict_adapter.py` | `pytorch/torchtitan` commit `7d563223d4ee10c0b8a7fc70d0cd119bcbcff93f` / blob `221fab1377ea753cb199f4d3de2f85ae91889014` | Implements bidirectional HF state-dict conversion for Kimi K3 multimodal + hybrid MLA/KDA layers, MXFP4 `weight_packed` load planning, KDA `A_log`/`dt_bias` `DTensor` layout preservation, and layer-0 unused AttnRes placeholder synthesis. |
| `torchtitan/models/deepseek_v3/state_dict_adapter.py` & `deepseek_v4/state_dict_adapter.py` | `pytorch/torchtitan` commit `1d581795cd014a46e871cc9fc605325fbbceee85` / blob `ac991872452b315d263ca36bc790cf9557557a84` | Filters `.weight_scale_inv` and out-of-range MTP layer keys from `fqn_to_index_mapping` and clones independent storage for `_MTP_SHARED_COPIES` in `to_hf()`. |
| `torchtitan/experiments/graph_trainer/simple_fsdp.py` | `pytorch/torchtitan` commit `3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0` / blob `3327d72b0708477e75e75c62241cb63b8fdffd18` | Disables active `SimpleFSDP` parametrizations via `disable_active_parametrization()` during `linear_param_shard_placements(model)` inspection on `meta` tensors. |

### Core Implementation Paths, Data Layout, and Synchronization Contracts

1. **Kimi K3 MXFP4 Dequantization & KDA `DTensor` Coordinate Transformations (`kimi_k3/state_dict_adapter.py:118-395`)**:
   - **MXFP4 Index Remapping**: The released Kimi K3 checkpoint stores routed experts in `compressed-tensors` MXFP4 format (`*.weight_packed` and `*.weight_scale`). Because `QuantizedHuggingFaceStorageReader(path, thread_count=4, target_dtype=torch.bfloat16)` dequantizes experts to BF16 on load and HF saves write unquantized BF16 tensors, `__init__` strips `_packed` from `fqn_to_index_mapping`, drops `.weight_scale`, and omits `vision_tower.*` / `mm_projector.*` when `vision_encoder is None`.
   - **KDA `A_log` Padding and `dt_bias` Flattening with `DTensor` Protection**: In the released MXFP4 checkpoint, KDA's `A_log` is zero-padded from `[n_heads]` to `[head_dim]` and `dt_bias` is flattened from `[n_heads, head_dim]` to `[n_heads * head_dim]`. Because padding and flattening alter tensor coordinates, `to_hf()` saves `(value.device_mesh, value.placements)` in `self._kda_parameter_layouts[key]` and redistributes `value` to `(Replicate(),) * ndim` before padding/flattening; `from_hf()` slices `value[:n_heads]` or reshapes to `[n_heads, head_dim]` and calls `value.redistribute(device_mesh=layout[0], placements=layout[1])`.
   - **Layer-0 AttnRes Placeholders**: `to_hf()` synthesizes `ones_like` / `zeros_like` placeholders for `language_model.model.layers.0.self_attention_res_{norm,proj}.weight` on the PP stage owning layer 0, while `from_hf()` drops `_UNUSED_HF_LAYER_ZERO_ATTN_RES_KEYS`.
2. **DeepSeek-V3 `91,991`-Entry HF Save Index Filtering & MTP Shared Weight Cloning (`deepseek_v3/state_dict_adapter.py:29-65, 248-290`)**:
   - **Why Unfiltered `fqn_to_index_mapping` Corrupts `safetensors`**: During DCP consolidation, `HuggingFaceStorageWriter` pre-allocates a header entry for every key in `fqn_to_index_mapping`. Any key not written by `to_hf()` receives `{"shape": [], "dtype": ""}`, causing `safetensors` to fail with `unknown variant ''` on reload.
   - Dropping `45,808` `.weight_scale_inv` keys and `788` `model.layers.61.*` MTP keys (when `num_mtp_layers == 0`) from the `91,991`-entry index yields an exact bijection with `to_hf()`: `45,395` entries for `num_mtp_layers=0` and `46,183` entries for `num_mtp_layers=1`.
   - **Independent MTP Storage Clones (`_MTP_SHARED_COPIES`)**: Because `safetensors` rejects aliased tensors sharing underlying storage within the same file, `to_hf()` clones `tok_embeddings.weight` and `lm_head.weight` into `model.layers.{61+i}.embed_tokens.weight` and `model.layers.{61+i}.shared_head.head.weight` (`3.707 GB` on 671B BF16), and `from_hf()` skips those duplicate keys.
3. **`GraphTrainer` `SimpleFSDP` Zero-Materialization Inspection on `meta` Device (`simple_fsdp.py:533-540`)**:
   - When `GraphPP` or `EP` has already parametrized submodules with `SimpleFSDP` (`ReplicateComputation`) while the model is still on the `meta` device, a subsequent `data_parallel(model)` call runs `linear_param_shard_placements(model)`, which reads `child.weight`.
   - Without `disable_active_parametrization()`, reading `child.weight` invokes `ReplicateComputation` and `_ShardedFSDPTensor._build_operands()`, triggering `dist_moe`'s MXFP8 check (`weight.is_cuda == False` $\to$ `ValueError: weight must be contiguous CUDA storage`). Wrapping `linear_param_shard_placements(model)` in `with disable_active_parametrization():` inspects the raw sharded parameter without materializing compute operands.

### Key Formulas, Units, and Engineering Assumptions

1. **DeepSeek-V3 HF Save Index Conservation Equation**:
   $$N_{\text{saved}}(L_{\text{mtp}}) = N_{\text{index\_total}} - N_{\text{fp8\_scale\_inv}} - \mathbb{I}(L_{\text{mtp}} == 0) \cdot N_{\text{mtp\_layer61}}$$
   With $N_{\text{index\_total}} = 91,991$, $N_{\text{fp8\_scale\_inv}} = 45,808$, and $N_{\text{mtp\_layer61}} = 788$, $N_{\text{saved}}(0) = 45,395$ and $N_{\text{saved}}(1) = 46,183$.
2. **MTP Shared Embedding/Head Clone Footprint**:
   $$\text{Bytes}_{\text{MTP\_clone}} = 2 \cdot L_{\text{mtp}} \cdot V \cdot H \cdot \text{sizeof}(\text{dtype}) = 2 \times 1 \times 129,280 \times 7,168 \times 2 = 3,707,232,256\text{ B} \approx 3.707\text{ GB}$$

### Synthetic Experiment and Verification Results (`experiments/torchtitan_kimi_k3_mxfp4_adapter_fsdp`)

Running `python experiments/torchtitan_kimi_k3_mxfp4_adapter_fsdp/run.py --check` verifies:

| Verification Dimension | Baseline / Unpatched Behavior | Patched Contract Behavior (`#5337` / `#5065` / `#5560`) | Deterministic Verification Result |
| :--- | :--- | :--- | :--- |
| Kimi K3 KDA `A_log` / `dt_bias` Round-Trip | Direct `pad`/`reshape(-1)` on sharded `DTensor` corrupts shard coordinates | Redistribute to `Replicate()`, pad `[64]->[128]` / flatten `8192`, restore `Shard(0)` in `from_hf()` | `a_log_exact_roundtrip = True`, `dtensor_layout_restored = True` |
| DeepSeek-V3 `safetensors` Index Filter (`mtp=0`) | Leaves `46,596` unwritten keys; consolidation emits `dtype: ""` headers | Retains exact `45,395` keys (`46,183` for `mtp=1`); `0` empty-dtype headers | `filtered_dcp_consolidation_empty_dtype_entries_mtp0 = 0` |
| DeepSeek-V3 MTP Shared Copies (`671B BF16`) | Shared storage alias rejected by `safetensors` writer | `.clone()` allocates independent storage (`3.707 GB`); skipped on `from_hf()` | Exact key set and storage isolation verified |
| `SimpleFSDP` Re-Applied `data_parallel()` (`meta`) | Pass 2 triggers `_build_operands()` $\to$ `weight must be contiguous CUDA storage` | `with disable_active_parametrization():` keeps `_build_operands_calls = 0` | `passes_completed = 2`, `error_raised = None` |

### Architect Design Exercises and Falsifiable Questions

1. **Debugging Exercise**: When exporting a fine-tuned BF16 checkpoint from a model initialized via an FP8/MXFP4 HuggingFace index, what causes `safetensors_rust.SafetensorError: unknown variant ''` on reload?
   - *Answer*: Unfiltered `.weight_scale_inv` / `.weight_scale` or omitted layer keys in `fqn_to_index_mapping` cause `HuggingFaceStorageWriter` consolidation to write placeholder entries with `{"shape": [], "dtype": ""}`.
2. **Falsifiable Hypothesis**: Can the `SimpleFSDP` `ValueError: weight must be contiguous CUDA storage` during `GraphPP` construction be fixed by calling `.contiguous()` on the weight before passing it to `dist_moe`?
   - *Verification*: False (falsified in CI run `39374680382`). Calling `.contiguous()` on a `meta` tensor still yields a `meta` tensor (`weight.is_cuda == False`); the inspection pass must disable active parametrizations via `disable_active_parametrization()`.
