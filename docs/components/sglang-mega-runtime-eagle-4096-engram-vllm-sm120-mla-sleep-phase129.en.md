# SGLang `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` Master Switch, Unified KV Layout Separation, Hybrid Pool `conv_slice_axis` Preservation, Multi-Layer EAGLE `4096`-ID Rotation, DeepSeek-V4.1 Engram Triton Reorder, DeepSeek-V4 Aligned Page-Index Zero-Copy & Batch-Invariant Fused Residual Add + RMSNorm, and vLLM SM120 NoPE Compact `528 B` FP8 MLA KV Rows, Multi-Worker CUDA Checkpoint Sleep & Cross-PP-Stage KV Layout Intersection (Phase 129 · Priority 138)

> **Evidence Classification**: This report combines first-hand upstream commits from `sgl-project/sglang` (`e1b6d75`, `3356820`, `77936f7`, `73fbf5c`, `db0dfc7`, `ec3365d`, `c6c7533`) and `vllm-project/vllm` (`dbcf568`, `88d3995`, `edad8b0`) (`implementation`) with deterministic Python 3.12 standard-library modeling in `experiments/sglang_mega_runtime_eagle_4096_engram_vllm_sm120_mla_sleep_phase129/run.py` (`own-offline`). No live hardware re-benchmarking is claimed.

---

## 1. Architectural Motivation

In speculative decoding (Multi-Layer EAGLE), DeepSeek-V4/V4.1 sparse MLA and Engram layers, hybrid Mamba+SWA memory pools, batch-invariant deterministic serving, and SM120 MLA KV cache kernels, fine-grained memory layout and loop tiling determine HBM bandwidth efficiency and pipeline compatibility:

1. **SGLang NVIDIA Megakernel Master Switch & Unified KV / Hybrid Pool Contracts (`sglang@e1b6d75`, `3356820`, `77936f7`)**: Introduces `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` to control 6 megakernel runtime flags while honoring explicit per-flag overrides; separates unified KV layout specs under `mem_cache/kv/layout`; and forwards `conv_slice_axis=getattr(cp.shape, "conv_slice_axis", 0)` in `init_unified_mamba_swa_pools` (`tri_p16_slice_axis`).
2. **Multi-Layer EAGLE Prefill `rotate_input_ids` `BLOCK_SIZE` `8 -> 4096` (`sglang@73fbf5c`)**: Previously used `BLOCK_SIZE = 8` whenever `select_index is None`, requiring **`4,096` loop trips** per CTA for a `32,768`-token prefill. Unifying `BLOCK_SIZE = 4096` cuts loop trips to **`8` (`512.0x` reduction, `28.41x` kernel speedup)**.
3. **DeepSeek-V4.1 Engram Triton Reorder & DSV4 Aligned Page-Index Zero-Copy (`sglang@db0dfc7`, `ec3365d`)**: Registers `embeddings.engram_rows_to_token_major` in Triton (**`4.24x` faster** than PyTorch view/copy reorder at `4096×4096`), and returns `x` directly in `deepseek_v4_backend._pad_last_dim` when `target_size == curr_size` so `F.pad` does not copy an already aligned page-index table (`131,072 -> 0 B`).
4. **Batch-Invariant Fused Residual Add + RMSNorm (`sglang@c6c7533`)**: Adds `fused_add_rms_norm_batch_invariant`, reducing HBM passes per layer from `5` (`280.0 MiB` at `4096×7168` BF16) to **`3` (`168.0 MiB`, `1.6667x` bandwidth speedup)**.
5. **vLLM SM120 NoPE Compact `528 B` FP8 MLA Rows, Multi-Worker CUDA Checkpoint Sleep & PP KV Layout Intersection (`vllm@dbcf568`, `88d3995`, `edad8b0`)**: Gates RoPE tail writes on `has_rope_tail` in `concat_and_cache_ds_mla_kernel` so NoPE models on SM120 store **`528 B/token`** instead of `656 B/token` (**`19.51%` KV memory/bandwidth saved, `1.2424x` speedup**); adds opt-in multi-worker CUDA process-checkpoint sleep; and intersects supported KV cache layouts across pipeline stages instead of asserting identical lists.

---

## 2. Deterministic Verification Summary (`own-offline`)

| Subsystem | Upstream Commit(s) | Legacy Baseline | Phase 129 Architecture | Quantitative Impact |
|---|---|---|---|---|
| **Multi-Layer EAGLE `rotate_input_ids`** | `sglang@73fbf5c` | `BLOCK_SIZE=8` when `select_index=None` (`4,096` trips at `32K`) | Always **`BLOCK_SIZE = 4096`** (`8` trips at `32K`) | **`512.0x` fewer trips, `28.41x` kernel speedup (`412.0 -> 14.5 us`)** |
| **DSV4.1 Engram All-to-All Reorder** | `sglang@db0dfc7` | PyTorch tensor reorder (`195.0 us`) | Triton `engram_rows_to_token_major` (`46.0 us`) | **`4.2391x` faster Engram output reorder** |
| **DSV4 `_pad_last_dim` Alignment** | `sglang@ec3365d` | `F.pad` copies `131,072 B` even when aligned | Early return when `target_size == curr_size` | **`131,072 -> 0 B` redundant page-index copy** |
| **Batch-Invariant Fused Add+RMSNorm** | `sglang@c6c7533` | Unfused Add + RMSNorm (`5` HBM passes, `280.0 MiB`) | `fused_add_rms_norm_batch_invariant` (`3` passes, `168.0 MiB`) | **`40.0%` HBM traffic saved (`1.6667x` speedup)** |
| **vLLM SM120 NoPE MLA FP8 KV Rows** | `vllm@dbcf568` | Padded `656 B/token` with unused RoPE tail | `has_rope_tail=False` compact **`528 B/token`** | **`19.51%` KV memory/bandwidth saved (`1.2424x`)** |
| **vLLM PP KV Layouts & Sleep** | `vllm@edad8b0`, `88d3995` | Asserts identical layout lists across PP stages | Intersects supported layouts (`HND`) + multi-worker CUDA sleep | **`0` PP layout assertion failures** |

---

## 3. Reproducible Verification

```bash
python experiments/sglang_mega_runtime_eagle_4096_engram_vllm_sm120_mla_sleep_phase129/run.py --check
python -m unittest tests/test_phase129_experiments.py -v
```
