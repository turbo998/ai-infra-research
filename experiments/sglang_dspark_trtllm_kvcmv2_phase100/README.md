# SGLang DSPARK FP8 LM Head, vLLM XPU Swapped-Grid MoE & TensorRT-LLM KVCMv2 Parallel Host Fill (Phase 100)

Deterministic Python 3.12 standard-library model reproducing:
1. **SGLang `#39622` (`01759f0a68`), `#40519` (`66cac6b79c`), and `#40435` (`589d196bc3`)**: AMD `gfx95` DSPARK verify-logits FP8 target `lm_head` GEMM via `torch._scaled_mm` (`1.85x` speedup at `num_verify_tokens=64`), Ascend NPU `DISPATCHER_QUANT_CONFIG` (`normal=bf16`, `low_latency=mxfp8`), and `arch35` lossless UE8M0 vs gated lossy FP32-block-to-MXFP8 requantization.
2. **vLLM `#51605` (`12d54c2948`) & `#63436` (`533906528c`)**: Intel XPU `batched_triton_kernel_swapped` grid-order transposition `(num_tiles, E)`, `int4_w4a16` `N = N * 2` shape resolution, and DFlash speculative decode `hidden_states.to(fc.params_dtype)` guard.
3. **TensorRT-LLM `#20790` (`cfae4519c3`)**: KVCMv2 `HostMem::startFill` parallel initial host-tier population (`1 + kInitialFillHelpers=7` threads in `4 MiB` calls before `HostMemBacking::commit(..., populated=true)`), achieving `6.4x` host pre-fault speedup.
