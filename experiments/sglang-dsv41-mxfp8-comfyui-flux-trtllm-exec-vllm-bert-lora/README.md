# SGLang DeepSeek-V4.1 Decode MXFP8 & Flux ControlNet, TRT-LLM Executor Cleanup & BOLT, and vLLM BERT Token-Classification LoRA

Deterministic Python 3.12 standard-library model reproducing:
- `sgl-project/sglang@028de2c5d4` (#45630) & `sgl-project/sglang@62b4127668` (#43698): DeepSeek-V4.1 `mxfp8_prefill_autotune_min_tokens = 1` (unlocking 14/14 decode/prefill buckets vs 2/14 at `>=4096`) and ComfyUI Flux ControlNet `_memo[id(value)]` residual deduplication + sequence-parallel sharding.
- `NVIDIA/TensorRT-LLM@a0141d0e56` (#20043) & `@00c184e1f8` (#20612): Removal of 35 obsolete `MpiTag` entries and 6 legacy executor compatibility files, plus `ENABLE_BOLT_PREMERGE_CONSUME = true` and nightly release BOLT side-by-side tarball publishing.
- `vllm-project/vllm@e8e43c6a09` (#61391), `@a0a645abda` (#62879), & `@c84edac4cb` (#62893): `TokenClassificationHeadWithLoRA` with per-request `AllPool(output_sizes_fn)` slicing (`hs[:, :size]`), shared-GPU NIXL prefill-before-decode launch serialization, and ROCm EAGLE3 `VLLM_BATCH_INVARIANT=1`.

## Run
```bash
python experiments/sglang-dsv41-mxfp8-comfyui-flux-trtllm-exec-vllm-bert-lora/run.py --check
```
