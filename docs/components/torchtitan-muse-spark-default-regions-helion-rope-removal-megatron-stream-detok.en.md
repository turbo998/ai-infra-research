# TorchTitan Muse Spark Default Compile Regions, Helion RoPE Removal / Non-Contiguous `ComplexRoPE` & Megatron-LM Streaming Detokenizer Alignment

**中文版**: [torchtitan-muse-spark-default-regions-helion-rope-removal-megatron-stream-detok.md](torchtitan-muse-spark-default-regions-helion-rope-removal-megatron-stream-detok.md)

## 1. Architectural Context & Problem Statement

In large-scale MoE training and dynamic inference serving (TorchTitan Muse Spark 816B-A42B, DeepSeek-V3 MTP, and Megatron-LM text generation), compiler region defaults, custom kernel dependencies, and streaming detokenization exhibited three bottlenecks:
1. **Recipe-Only Muse Spark Compile Regions**: `MuseSparkModel` defined `muse_spark_input_scaled_rmsnorm`, `muse_spark_logit_postprocess`, and `muse_spark_residual`, but only the Dist-MoE scale recipes added them to `local_compile_regions`. Base, debug, and RL DAPO recipes ran eager kernels costing **`2,440 us` per layer at 4k tokens** (`724 us` residual, `381 us` post-FFN norm, `1,335 us` expert-output norm) against a **`331 us` bandwidth ideal (`13.57%` efficiency)**.
2. **Obsolete CUDA-Only Helion RoPE Dependency**: With compiled `ComplexRoPE` matching or beating Helion up to 128k context, `torchtitan_recipes/overrides/helion_rope.py` (`1,031` lines + `helion==1.1.0`) had zero production callers yet forced `skip_rocm_test=True` on `deepseek_v3_mtp_fsdp+ep`.
3. **Megatron-LM Streaming vs Non-Streaming Divergence**: `HuggingFaceFastIncrementalDetokenizer` used `DecodeStream(skip_special_tokens=True)`, stripping every `AddedToken(..., special=True)`—including `<think>`, `</think>`, `<tool_call>`, and `</tool_call>`—whereas non-streaming decoding only stripped `all_special_ids` and trailing EOS tokens.

## 2. First-Hand Implementation Audit

### 2.1 `pytorch/torchtitan` (`39bd2054d9c4fcbcabc343d262abad85f19020fb`, `#5433`, `2026-10-20`)
- **`torchtitan/models/muse_spark/model.py` & `torchtitan_recipes/models/muse_spark.py`**:
  - Promotes `muse_spark_input_scaled_rmsnorm`, `muse_spark_logit_postprocess`, and `muse_spark_residual` into `MuseSparkModel.Config.local_compile_regions` by default (6 regions total; LoRA SFT still overrides to `["loss"]`).
  - **Author Benchmark (`author-benchmark`, GB300 2-layer 816B-A42B EP32)**:
    - Training (2k/4k/16k): **`-6.9%` to `-10.0%` vs PR 1** (**`-13.6%` to `-16.6%` vs `main`**, e.g., 4k CUDA graphs `42.13 -> 39.28 -> 35.94 ms`);
    - Prefill (2k/16k): **`-10.0%` to `-10.9%` vs PR 1** (**`-14.6%` to `-15.7%` vs `main`**);
    - Decode (1/8/64/256 seqs): **`-7.1%` to `-11.2%` vs PR 1** (256 seqs `1.55 -> 1.49 -> 1.32 ms`, **`-14.6%` vs `main`**), staying within `4` residual and `6` norm graphs (`limit=8`).

### 2.2 `pytorch/torchtitan` (`abdf14320a01212430b4ddfb2d147a4dfbab4b3e`, `#5530`, `2026-10-20`)
- **`tests/unit_tests/gpu/models/qwen3_5/test_rope_compile.py` & `tests/integration_tests/models.py`**:
  - Deletes `helion_rope.py` and `helion==1.1.0`, enables ROCm on `deepseek_v3_mtp_fsdp+ep`, and ports non-contiguous coverage (`query = query_leaf.transpose(0, 1)` with strides `(64, 4096, 1)` and `key = key_leaf[..., -64:].unsqueeze(1)` with strides `(96, 64, 1)`) to `test_rope_compile.py` without the contiguous `int32` shortcut.

### 2.3 `NVIDIA/Megatron-LM` (`ebab8e2e21c77361fd9edbaee3e3d7d90748f340`, `#8268`, `2026-10-20`)
- **`megatron/core/inference/text_generation_server/dynamic_text_gen_server/incremental_detokenizer.py`**:
  - Replaces `self._skip_special_tokens = True` with `self._skipped_token_ids = frozenset(huggingface_tokenizer.all_special_ids) | model_eos_token_ids(tokenizer)`, preserving reasoning and tool-call markers across streaming deltas.

## 3. Deterministic Verification (`own-offline`)

Run `python experiments\torchtitan_muse_default_regions_rope_megatron_detok\run.py --check`:

| Subsystem | Metric | Before / Baseline | After (`Phase 37`) | Evidence Tier |
|---|---|---|---|---|
| Muse Spark Default Regions | 4k Per-Layer Eager vs Ideal | `2,440.0 us` (`13.57%` eff.) | **6 regions default (`331.0 us` ideal)** | `author-benchmark` + `own-offline` |
| Muse Spark GB300 2L Latency | 4k Train Graphs / Decode-256 | `42.13 ms` / `1.55 ms` (`main`) | **`35.94 ms` (`-14.7%`) / `1.32 ms` (`-14.6%`)** | `author-benchmark` |
| TorchTitan ComplexRoPE | Non-Contiguous `(64,4096,1)` & `(96,64,1)` | `helion==1.1.0` (`skip_rocm=True`) | **`-1,031` lines, ROCm enabled, `0.0` diff** | `implementation` + `own-offline` |
| Megatron Stream Detok | `<think>plan</think>answer<tool_call>call</tool_call>` | `"plananswercall"` (markers dropped) | **Exact match with non-streaming output** | `implementation` + `own-offline` |
