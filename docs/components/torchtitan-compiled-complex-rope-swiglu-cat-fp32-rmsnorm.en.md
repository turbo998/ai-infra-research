# TorchTitan Compiled Real-Arithmetic `ComplexRoPE` (`int32` `bf16` Pair Packing), Unmasked SwiGLU Backward Stores, and Muse Spark `fp32` RMSNorm Fusion

**Language / 语言**: [English](torchtitan-compiled-complex-rope-swiglu-cat-fp32-rmsnorm.en.md) | [中文](torchtitan-compiled-complex-rope-swiglu-cat-fp32-rmsnorm.md)

> **Evidence Tier**: `implementation` (first-hand upstream commit inspection) + `author-benchmark` (TorchTitan author measurements on NVIDIA GB300 / A10G / MI350X) + `own-offline` (Python 3.12 standard-library deterministic verification via `experiments/torchtitan_compiled_rope_swiglu_fp32_rmsnorm/run.py --check`).

---

## 1. Executive Summary & Architectural Findings

In PyTorch `torch.compile` (TorchInductor) `@local_compile` regions, complex-dtype fallbacks, masked concatenation pointwises, and mixed-precision input/weight dtype mismatches frequently degrade memory-bound layers into multi-kernel eager fallbacks or 2x bandwidth penalties. Three 2026-10-20 commits in `pytorch/torchtitan` establish concrete compiler co-design patterns:

1. **Compiled Real-Arithmetic `ComplexRoPE` with Contiguous `int32` `bf16`-Pair Packing ([`pytorch/torchtitan@b95271ed07`](https://github.com/pytorch/torchtitan/commit/b95271ed0709d63a60d185b88994cba10ea748a1))**:
   - **Problem**: Inductor cannot generate kernels for `view_as_complex` and complex multiplication, forcing `ComplexRoPE` to fall back to separate eager kernels (`1126.5 us` Llama 3 8B 64k prefill and `2589.9 us` DeepSeek-V3 671B 64k prefill on GB300).
   - **Float Cache Gather & Real Rotation**: Under `torch.compiler.is_compiling()`, `_reshape_cache` gathers rows from the real view `torch.view_as_real(self.cache).flatten(-2)` (`(T, 1, dim)`) so the gather fuses directly into `_RotatePairs.rotate`.
   - **Contiguous `int32` Read/Write for `bf16` Pairs**: Separate per-half stores write every other element (`stride-2`), failing to vectorize at ragged dynamic-`T` row tails. `_RotatePairs.rotate` views contiguous `bf16` inputs as `int32`, unpacks `x_real = (pairs << 16).view(torch.float32)` and `x_imag = (pairs & ~0xFFFF).view(torch.float32)`, rotates in `fp32`, and writes each rotated `bf16` pair as one `int32` (`(low | high).view(torch.bfloat16)`). On GB300 (`author-benchmark`), 64k prefill drops from `1126.5 -> 224.3 us` (**`5.02x` speedup**, beating the `231.5 us` Helion kernel) on Llama 3 8B and from `2589.9 -> 426.5 us` (**`6.07x` speedup**) on DeepSeek-V3 671B (where Helion fails at 128k due to `int32` index overflow).
   - **Deterministic Pointwise Lowering**: Setting `"triton.autotune_pointwise": False` prevents Inductor from timing-selecting kernel configs with different FMA contractions, collapsing `9` distinct step-1 losses across 13 runs of `deepseek_v3_fsdp+cp+pp+ep` down to **`1` identical loss across 12 runs**.

2. **SwiGLU Backward Unmasked Per-Half Stores ([`pytorch/torchtitan@eb48bab350`](https://github.com/pytorch/torchtitan/commit/eb48bab35099a79dfcd7d9c5d26b5af5e41f06d8))**:
   - **Problem**: `fused_binary_activation` unbinds `gate_up [..., 2, F]`, so backward stacks `grad_gate` and `grad_up` into `[..., 2, F]`. Inductor's default cat lowering emits a single pointwise kernel over both halves that loads inputs behind a mask and selects a half with `tl.where` (`788 us` backward at `T=16k, F=2048` on GB300, `2.09x` its `377 us` bandwidth floor).
   - **Solution**: Passing both `max_pointwise_cat_inputs=0` and `max_complex_pointwise_cat_inputs=0` (setting only the first leaves `max_complex_pointwise_cat_inputs=8` routing a 2-input cat down the masked path) makes Inductor emit one unmasked store per half. On GB300 (`author-benchmark`), `T=16k, F=2048` SwiGLU `fwd+bwd` drops from `1440.0 -> 1047.8 us` (**`-27.24%`**, `1.20%` faster than the `1060.5 us` hand-written Triton override).

3. **Muse Spark `fp32`-Stream RMSNorm Fused Kernel Restoration ([`pytorch/torchtitan@b78c4aed96`](https://github.com/pytorch/torchtitan/commit/b78c4aed9640b5fb034b9be1c05f24fe30f1d093))**:
   - Muse Spark keeps its residual stream in `fp32` while `effective_weight_D` (`weight + gain_center`) is `bf16`. `F.rms_norm` dispatches its fused kernel only when `input.dtype == weight.dtype`, otherwise falling back to `~20` small ATen kernels (`pow`, `mean`, `rsqrt`, `mul`). Pre-casting `effective_weight_D = effective_weight_D.to(input_dtype)` restores the single fused forward kernel (`20 -> 1` kernel launch, `-7%` 2-layer training time, `-5%` prefill, `-2%..-4%` decode) with `0.0` numerical drift.

---

## 2. GB300 Benchmark & Deterministic Verification Summary (`author-benchmark` + `own-offline`)

Verified via `python experiments/torchtitan_compiled_rope_swiglu_fp32_rmsnorm/run.py --check` (`experiments/torchtitan_compiled_rope_swiglu_fp32_rmsnorm/results/summary.json`):

| Module | Workload (`GB300`, `author-benchmark`) | Baseline (`main` / Eager) | Optimized (`this PR`) | Custom Kernel Reference | Key Finding (`own-offline` + `author-benchmark`) |
|---|---|---:|---:|---:|---|
| **`ComplexRoPE` (`rope` region)** | Llama 3 8B (`T=64k` prefill) | `1126.5 us` | **`224.3 us`** | `231.5 us` (Helion) | **`5.02x`** speedup, `0` ULP drift on `int32` pair packing |
| **`ComplexRoPE` (`rope` region)** | DeepSeek-V3 671B (`T=64k` prefill) | `2589.9 us` | **`426.5 us`** | `384.9 us` (Helion fails @ 128k) | **`6.07x`** speedup, `9 -> 1` step-1 loss variance across runs |
| **SwiGLU (`fused_binary_activation`)** | `[8T, 2, 2048]` (`T=16k` train fwd+bwd) | `1440.0 us` | **`1047.8 us`** | `1060.5 us` (Triton override) | **`-27.24%`**, `1.20%` faster than Triton override (`1.05x` floor) |
| **Muse Spark `RMSNorm`** | `fp32` residual stream + `bf16` weight | `~20` fallback kernels | **`1` fused fwd kernel** | ATen fused `rms_norm` | `-7.0%` 2-layer train, `-5.0%` prefill, `0.0` numerical diff |

---

## 3. Architect Design Exercise & Falsifiable Follow-Up

1. **Bitwise Packing Exercise**: Why does `(pairs << 16).view(torch.float32)` and `(pairs & ~0xFFFF).view(torch.float32)` reconstruct exact `fp32` values from a little-endian `int32` holding two `bf16` elements `(x[2i], x[2i+1])`?
   - **Answer**: Because `bf16` has the exact same 1-bit sign and 8-bit exponent layout as `fp32` in bits `[31:16]`, shifting the low 16-bit `x[2i]` left by 16 bits or masking out the low 16 bits for the high 16-bit `x[2i+1]` produces exact IEEE-754 `fp32` bit patterns with zero mantissa loss.
2. **Falsifiable Follow-Up**: Why do `max_pointwise_cat_inputs=0` and `max_complex_pointwise_cat_inputs=0` slow down `CosSinRoPE` (split-half layout) by `1.3x-2.6x` on Qwen3 shapes while accelerating `ComplexRoPE` (`fp32`/`fp16` stack) by `1.4x`?
