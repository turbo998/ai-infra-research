# SGLang GLM-5.2 Compact DSA Indexer & Dynamic INT8 MLAProlog and Megatron-LM Meta-Device Tied Embedding Sync

Deterministic Python 3.12 standard-library model verifying:
1. **`sgl-project/sglang` `3bf4cbc6cf18b539d88c08c17cb97d82ec82c951` (`#44436`)**: Compact DSA indexer allocation for hybrid `"full"` / `"shared"` indexer layers on non-arch35 NPUs, cutting indexer memory by 75% (at 15/60 full layers) and expanding KV token capacity.
2. **`sgl-project/sglang` `10d5293546faf6469ebe7b924ecf5d9519e14c6f` (`#43469`)**: Dynamic INT8 `MLAProlog` kernel-class dispatch (`weight_quant_mode == 2`), `w_kc` storage retention in disaggregated decode, and selective BF16 indexer `q_lora` recomputation.
3. **`NVIDIA/Megatron-LM` `fa322ce62077fa34ac6927fdc66ab2c8b88fcc30` (`#8010`)**: `_materialize_shared_embedding_weight()` before `embd_group` `all_reduce` under `init_model_with_meta_device=True`, including `torch.zeros_like` NaN prevention for checkpoint-loaded owners and attribute preservation across TE parameter replacement.

## Run
```bash
python experiments/sglang_glm52_dsa_mlaprolog_megatron_meta_embed/run.py
python experiments/sglang_glm52_dsa_mlaprolog_megatron_meta_embed/run.py --check
```
