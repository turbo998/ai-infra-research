# `torch.compile` (`TorchInductor` / `FxGraphCache` / `AOT`) Cold-Start Elimination & Operator Fusion

Deterministic offline model quantifying production LLM serving startup latency (`Llama-3-70B-80L-TP8-FP8` across `32` CUDA Graph capture batch buckets) and pointwise operator fusion Roofline gains under PyTorch `TorchInductor` and vLLM V1 compilation backends.

## Primary Code References

- **`vllm-project/vllm` (`commit 82a24bb82dd9344cce7a43195bb51a2390adf00a`)**:
  - `vllm/compilation/wrapper.py` (`blob 2f8c7ce2bac3`): `TorchCompileWithNoGuardsWrapper` (`guard_filter_fn = torch.compiler.skip_all_guards_unsafe` and `set_stance("fail_on_recompile")`).
  - `vllm/compilation/caching.py` (`blob 5205c7404fcb`): `StandaloneCompiledArtifacts` two-level SHA256 content-addressed deduplication (`submodule_bytes_store[hex_digest]`), collapsing 80 identical attention-split layers into `2` unique compiled artifacts.
  - `vllm/compilation/backends.py` (`blob d45be552a7c9`): `VllmBackend` piecewise compilation manager and `MEGA_AOT_ARTIFACT` loading.
- **`pytorch/pytorch` (`commit 1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340`)**:
  - `torch/_inductor/codecache.py` (`blob de72c7e8e7680aae23baa4ffc588425d35110845`) & `compile_fx.py` (`blob 92b74aae85308a970a8c7e6daca547853edbc8c1`): `FxGraphCache` (`_filter_backed_symints`, `load_with_key`) and Triton autotune cache.

## Key Results (`results.json`)

- **Cold-Start Elimination (`29.44 min -> 18.0 s / 11.8 s`)**:
  - Unshared per-bucket `torch.compile` traces `32` dynamic batch buckets without layer deduplication (`3,072` Triton autotune benchmarks), blocking server startup for **`1,766.2 s` (`29.44 min`)** and suffering **`19`** runtime guard-miss recompilations.
  - vLLM single symbolic trace (`TorchCompileWithNoGuardsWrapper`) + `StandaloneCompiledArtifacts` SHA256 layer deduplication cuts cold compilation to **`69.0 s`** (`25.60x` faster, `0` runtime recompiles).
  - Pre-warmed `FxGraphCache` + Triton Autotune Cache reduces startup to **`18.0 s`** (**`98.12x`** speedup), and `AOTInductor` (`MEGA_AOT_ARTIFACT`) reaches **`11.8 s`** (**`149.68x`** speedup).
- **Pointwise Operator Fusion Roofline**:
  - Fusing `4` memory-bound subgraphs per layer (`Pre/Post-Attn Residual + RMSNorm + FP8 Dynamic Quant`, `In-place QK RoPE`, `SwiGLU + FP8 Dynamic Quant`) collapses `1,120` eager pointwise kernels across 80 layers into `320` Triton kernels, cutting pointwise HBM round-trip traffic by **`56.2%`** (`6,050 MiB -> 2,650 MiB` at `B=256`) and increasing end-to-end serving throughput by **`+35.73%`** (`12,571.9 -> 17,063.9 tok/s`).

## Reproduction

```bash
python experiments/torch_compile_inductor_cache/run.py
python experiments/torch_compile_inductor_cache/run.py --check
```
