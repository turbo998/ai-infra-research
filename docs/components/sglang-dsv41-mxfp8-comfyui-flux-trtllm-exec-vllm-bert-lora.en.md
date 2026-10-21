# SGLang DeepSeek-V4.1 Decode MXFP8 Autotuning & ComfyUI Flux ControlNet Residual Sharding, TensorRT-LLM Legacy Executor Removal & Nightly BOLT Pipeline, and vLLM BERT Token-Classification LoRA & Shared-GPU NIXL Staging

> **Updated**: 2026-10-22 · **Phase**: Phase 53 (`priority: 62`) · **Category**: `inference-engine`  
> **Deterministic Experiment**: [`experiments/sglang-dsv41-mxfp8-comfyui-flux-trtllm-exec-vllm-bert-lora/run.py`](../../experiments/sglang-dsv41-mxfp8-comfyui-flux-trtllm-exec-vllm-bert-lora/run.py)

---

## 1. First-Hand Upstream Sources

| Repository & Commit | Date (UTC) | Core Engineering Contribution |
|---|---|---|
| [`sgl-project/sglang@028de2c5d4`](https://github.com/sgl-project/sglang/commit/028de2c5d485385bee3b68f6463a0a887f42e4d2) | 2026-10-21 | `[Perf] dsv41`: Lower `mxfp8_prefill_autotune_min_tokens` from `4096` to `1` in DeepSeek-V4.1 FlashInfer CuTeDSL MXFP8 layers (#45630) |
| [`sgl-project/sglang@62b4127668`](https://github.com/sgl-project/sglang/commit/62b4127668e6b4f3a8b8617a34b60679695a8f6f) | 2026-10-21 | `[Feat] comfyui`: Deduplicated ControlNet double/single-block residual packing, sequence-parallel (`shard_latents_for_sp`) sharding, and `_SESSION_SKIP_EXTRA` isolation in Flux integrated mode (#43698) |
| [`NVIDIA/TensorRT-LLM@a0141d0e56`](https://github.com/NVIDIA/TensorRT-LLM/commit/a0141d0e56af8fad6df709151f78c8769f47335d) | 2026-10-21 | `[TRTLLM-17060][refactor]`: Remove 35 legacy `MpiTag` enum entries and 6 obsolete executor compatibility files (#20043) |
| [`NVIDIA/TensorRT-LLM@00c184e1f8`](https://github.com/NVIDIA/TensorRT-LLM/commit/00c184e1f81e05ca55b22494c400f9a48d1811ee) | 2026-10-21 | `[None][fix]`: Enable `ENABLE_BOLT_PREMERGE_CONSUME = true` and match nightly release BOLT variant publishing to post-merge (#20612) |
| [`vllm-project/vllm@e8e43c6a09`](https://github.com/vllm-project/vllm/commit/e8e43c6a09e4a5fd047cf02b5923db9b2d88ded9) | 2026-10-21 | `[LoRA]`: Add `TokenClassificationHeadWithLoRA` and dynamic `AllPool(output_sizes_fn)` slicing for BERT token-classification adapters (#61391) |
| [`vllm-project/vllm@a0a645abda`](https://github.com/vllm-project/vllm/commit/a0a645abdad134bafbc1a8954dd436e4fb0a5e61) | 2026-10-21 | `[CI][KV Connector]`: Wait for NIXL prefill instances before starting decode instances when `GPUS_NEEDED > num_gpus` (#62879) |
| [`vllm-project/vllm@c84edac4cb`](https://github.com/vllm-project/vllm/commit/c84edac4cba45476755f9c462425ffc8290ab6aa) | 2026-10-21 | `[ROCm][CI]`: Enable `VLLM_BATCH_INVARIANT=1` for ROCm EAGLE3 speculative decoding accuracy tests (#62893) |

---

## 2. Deep-Dive Architectural Analysis

### 2.1 SGLang DeepSeek-V4.1 Decode MXFP8 Tactics & Flux ControlNet Residuals (`028de2c5d4`, `62b4127668`)

- **DeepSeek-V4.1 MXFP8 Decode Autotuning (`028de2c5d4`)**: Lowers `method.mxfp8_prefill_autotune_min_tokens` from `4096` to `1` in `deepseek_v4.py` when `backend.is_flashinfer_cutedsl()` and `method.block_fp8_as_mxfp8` are active, unlocking tuned MXFP8 tactics across all **14/14** decode and prefill token buckets (`1..8192`) instead of only `2/14` (`>=4096`).
- **ComfyUI Flux ControlNet Residual Deduplication & SP Sharding (`62b4127668`)**: Deduplicates repeated ControlNet residual tensors across Flux's 19 double-blocks and 38 single-blocks via object-identity memoization (`_memo[id(value)]`, `detached[id(residual)]`, `prepared[id(residual)]`), cutting IPC pickle and sequence-parallel (`shard_latents_for_sp`) preparation overhead by **`5.7x`**, while adding `"comfyui_control"` to `_SESSION_SKIP_EXTRA` so control residuals never leak across denoising steps.

### 2.2 TensorRT-LLM Legacy Executor Cleanup & Nightly BOLT Variant Publishing (`a0141d0e56`, `00c184e1f8`)

- **Legacy Executor Compatibility Removal (`a0141d0e56`)**: Strips **35 obsolete `MpiTag` values** (`36 -> 1`, leaving only `MpiTag::kDefault = 0`) and removes 6 legacy executor files (`requestWithId.{h,cpp}`, `requestUtils.{h,cpp}`, `orchestratorUtils.h`, `cache_transmission/nixl_utils/interfaces.h`).
- **Nightly Release BOLT Alignment (`00c184e1f8`)**: Flips `ENABLE_BOLT_PREMERGE_CONSUME = true` and gates `BOLT_PUBLISH_VARIANT` on `isBoltReleasePipeline() && ENABLE_BOLT_POSTMERGE_VARIANT`, keeping the canonical uninstrumented tarball intact for `BoltProfileGen` while publishing the BOLT-optimized tarball alongside it.

### 2.3 vLLM BERT Token-Classification LoRA, Shared-GPU NIXL & ROCm EAGLE3 (`e8e43c6a09`, `a0a645abda`, `c84edac4cb`)

- **`TokenClassificationHeadWithLoRA` (`e8e43c6a09`)**: Extends `BaseClassificationHeadWithLoRA` with `get_output_num_labels()` and wires `AllPool(output_sizes_fn)` in `TokenPooler` to slice each request's token logits via `hs[:, :size]`, returning exact per-adapter label dimensions (`[[12, 3], [8, 9], [15, 5]]` under `max_lora_cls_labels=16`).
- **Shared-GPU NIXL Serialization (`a0a645abda`) & ROCm EAGLE3 Batch Invariance (`c84edac4cb`)**: Serializes NIXL decoder startup behind `wait_for_server "$PORT"` when `GPUS_NEEDED > num_gpus` so decoder KV memory profiling does not count concurrent prefiller weight loading, and sets `VLLM_BATCH_INVARIANT=1` on ROCm EAGLE3 speculative decoding to prevent `hipBLASLt` batch-shape tactic shifts.
