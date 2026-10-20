# Deep Dive: SGLang GLM-5.2 Compact DSA Indexer Allocation, Dynamic INT8 MLAProlog Dispatch, and Megatron-LM Meta-Device Tied Embedding Synchronization

> **中文版**: [sglang-glm52-compact-dsa-int8-mlaprolog-megatron-meta-tied-embed.md](sglang-glm52-compact-dsa-int8-mlaprolog-megatron-meta-tied-embed.md) | **Back to Index**: [README.en.md](../../README.en.md) | **7-Card Benchmark Report**: [ai-infra-phase33-benchmark-cards.html](../reports/ai-infra-phase33-benchmark-cards.html)

## 1. First-Hand Source Matrix (2026-10-20)

| Component | Repository & Commit SHA | Release Date | Key Files & Symbols | Evidence Class |
| :--- | :--- | :--- | :--- | :--- |
| **SGLang GLM-5.2 Compact DSA Indexer Pool Allocation** | `sgl-project/sglang` [`3bf4cbc6cf18b539d88c08c17cb97d82ec82c951`](https://github.com/sgl-project/sglang/commit/3bf4cbc6cf18b539d88c08c17cb97d82ec82c951) (`#44436`) | `2026-10-20` | `python/sglang/srt/mem_cache/kv_cache_configurator.py::_build_ascend_mla_kv_pool`, `python/sglang/srt/model_executor/pool_configurator.py::_compute_dsa_indexer_cell_size` | `implementation` + `own-offline` |
| **SGLang GLM-5.2 Dynamic INT8 MLAProlog & BF16 Indexer Adaptation** | `sgl-project/sglang` [`10d5293546faf6469ebe7b924ecf5d9519e14c6f`](https://github.com/sgl-project/sglang/commit/10d5293546faf6469ebe7b924ecf5d9519e14c6f) (`#43469`) | `2026-10-20` | `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py::npu_mla_preprocess` | `implementation` + `own-offline` |
| **Megatron-LM Tied Embedding Materialization & Sync under Meta-Device Init** | `NVIDIA/Megatron-LM` [`fa322ce62077fa34ac6927fdc66ab2c8b88fcc30`](https://github.com/NVIDIA/Megatron-LM/commit/fa322ce62077fa34ac6927fdc66ab2c8b88fcc30) (`#8010`) | `2026-10-20` | `megatron/core/models/common/language_module/language_module.py::LanguageModule._materialize_shared_embedding_weight` | `implementation` + `own-offline` |

---

## 2. SGLang GLM-5.2-W4A8C8 Compact DSA Indexer Pool Allocation

### 2.1 Redundant Indexer Slot Allocation Across `"full"` and `"shared"` DSA Layers
In hybrid Dynamic Sparse Attention (DSA) models such as GLM-5.2, `hf_config.indexer_types` alternates between `"full"` layers (which compute and store `index_k_with_scale_buffer`) and `"shared"` layers (which reuse an upstream `"full"` layer's sparse indices and never touch their own indexer buffer).

Prior to Commit `3bf4cbc6cf18`, `_build_ascend_mla_kv_pool` gated `use_compact_indexer_layout` on `is_arch35`, and `_compute_dsa_indexer_cell_size` forced `if not is_npu_arch35(): allocate_all_layers = True`. Consequently, non-arch35 NPUs allocated indexer memory for all layers—including `"shared"` layers that never access it—shrinking the available MLA KV cache budget.

### 2.2 Quantitative Capacity Gain After Removing the `is_arch35` Gate
Commit `3bf4cbc6cf18` removes both `is_arch35` gates so pool sizing and allocation uniformly size only active `"full"` indexer layers:
- For a 60-layer GLM-5.2 configuration (15 `"full"` layers, 45 `"shared"` layers, MLA KV dim $576$, indexer head dim $128$, 2-byte BF16 storage):
  - Per-token indexer memory drops from $60 \times 128 \times 2 = 15{,}360\,\text{B}$ to $15 \times 128 \times 2 = 3{,}840\,\text{B}$ (**$75.0\%$ reduction**).
  - Total per-token KV+Indexer cell footprint drops from $84{,}480\,\text{B}$ to $72{,}960\,\text{B}$, expanding maximum KV token capacity under a $24.0\,\text{GiB}$ pool budget from **$305{,}040$ to $353{,}204$ tokens ($+15.79\%$)** (and from `118` to `134` tokens in the 6-layer unit-test geometry).

---

## 3. SGLang GLM-5.2 Dynamic INT8 `MLAProlog` Dispatch & BF16 Indexer Adaptation

Commit `10d5293546fa` (`#43469`) resolves three production hazards when enabling fused `MLAProlog` for dynamic INT8 (`W8A8`) on GLM-5.2:
1. **Kernel-Class Inspection Instead of Ambiguous `int8` Dtype**: Because both static and dynamic W8A8 store `int8` weights, `mla_preprocess.py` inspects the quantization kernel instances on `(self.qkv_a_proj, self.q_b_proj)`, setting `weight_quant_mode = 2` only when both use `NPUW8A8Int8DynamicLinearMethod` (and `3` for `NPUMXFP8LinearMethod`).
2. **Retaining `w_kc` in Disaggregated Decode**: Prevents `m.w_kc.untyped_storage().resize_(0)` when `weight_quant_mode == 2`.
3. **Selective BF16 Indexer `q_lora` Recomputation**: When `weight_quant_mode == 2` but `m.indexer.wq_b` is not using `NPUW8A8Int8DynamicLinearMethod`, `npu_mla_preprocess` recomputes unquantized `q_lora = m.q_a_layernorm(m.fused_qkv_a_proj_with_mqa(hidden_states)[0][..., :m.q_lora_rank])` with `dynamic_scale = None`.

---

## 4. Megatron-LM Tied Embedding Synchronization Under Meta-Device Init (`#8010`)

Commit `fa322ce62077` (`#8010`) fixes `LanguageModule.setup_embeddings_and_output_layer` when `init_model_with_meta_device=True` and `share_embeddings_and_output_weights=True`:
- Calls `_materialize_shared_embedding_weight()` before the `embd_group` `all_reduce` and removes the `and not self.config.init_model_with_meta_device` guard that previously skipped synchronization.
- Uses `torch.empty_like` followed by `owner.reset_parameters()` when the owner module defines an initializer (e.g., a Transformer Engine LM head on the last pipeline stage), or **`torch.zeros_like`** when the owner has no initializer (checkpoint-load workflow) so uninitialized memory containing `NaN`s can never corrupt the `embd_group` `all_reduce`.
- Migrates custom Megatron parameter attributes (`vars(old_param)`) across TE `Parameter` object replacement during `reset_parameters()`.

---

## 5. Reproducible Experiment

```bash
python experiments/sglang_glm52_dsa_mlaprolog_megatron_meta_embed/run.py --check
```
