# Experiment: TorchTitan Compiled Real-Arithmetic `ComplexRoPE`, Unmasked SwiGLU Backward Cat & Muse Spark `fp32` RMSNorm Fusion (`torchtitan_compiled_rope_swiglu_fp32_rmsnorm`)

Deterministic Python 3.12 standard-library verification for Phase 34 (`priority: 43`) Topic 2:
- **`pytorch/torchtitan@b95271ed07`**: `ComplexRoPE` compiled real-arithmetic `_RotatePairs` with contiguous `int32`-packed `bf16` pairs (`(pairs << 16).view(float32)`, `(pairs & ~0xFFFF).view(float32)`, `0` ULP drift vs reference complex multiply, `5.02x` Llama 3 8B 64k prefill speedup, `6.07x` DeepSeek-V3 671B 64k prefill speedup) and `triton.autotune_pointwise=False` (`9 -> 1` distinct step-1 loss across multi-run FSDP+CP+PP+EP).
- **`pytorch/torchtitan@eb48bab350`**: `fused_binary_activation` SwiGLU backward with `max_pointwise_cat_inputs=0` and `max_complex_pointwise_cat_inputs=0`, replacing masked `tl.where` cat stores with unmasked per-half stores (`1440.0 -> 1047.8 us`, `-27.24%` on GB300 at `T=16k, F=2048`, `1.20%` faster than hand-written Triton override).
- **`pytorch/torchtitan@b78c4aed96`**: `MuseSparkRMSNorm.forward` pre-casting `bf16` `effective_weight_D` to `input.dtype` (`fp32`) so `F.rms_norm` dispatches its single fused kernel instead of `~20` decomposed ATen kernels (`0.0` numerical drift, `-7%` 2-layer training time).

## Run
```bash
python experiments/torchtitan_compiled_rope_swiglu_fp32_rmsnorm/run.py --check
```
