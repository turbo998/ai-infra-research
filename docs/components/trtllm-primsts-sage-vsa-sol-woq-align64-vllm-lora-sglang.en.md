# Deep Dive into TensorRT-LLM PrimsTS SageAttention Block-Sparse VSA/SOL, INT8-woq MoE 64-Row Tile Padding, vLLM LoRA MoE `_lora_context` Reload Rebinding, and SGLang Rust PD Bootstrap Port Routing

> **Language / 语言**: [English](trtllm-primsts-sage-vsa-sol-woq-align64-vllm-lora-sglang.en.md) | [中文](trtllm-primsts-sage-vsa-sol-woq-align64-vllm-lora-sglang.md)  
> **Phase**: 71 (`priority: 80`, Track B — LLM Serving, Quantization & Disaggregated PD)  
> **Evidence Tier**: `implementation` (pinned-commit source audit) + `author-benchmark` (author B200 SageAttention INT8/FP8 accuracy measurements) + `own-offline` (deterministic stdlib verification `experiments/trtllm-primsts-sage-vsa-sol-woq-align64-vllm-lora-sglang/run.py`)

---

## 1. Architectural Context & Core Engineering Questions

Recent commits across TensorRT-LLM, vLLM, and SGLang address four critical production serving challenges spanning block-sparse quantized attention, tensor-parallel MoE tile alignment, LoRA checkpoint reload, and disaggregated prefill-decode control planes:

1. **TensorRT-LLM PrimsTS SageAttention for Block-Sparse VSA and SOL (`NVIDIA/TensorRT-LLM@791a402fc8d5`, `#20224`)**: How does VisualGen integrate SageAttention (`smooth_k`, `smooth_v`, sub-tile `k_block_size` $\in \{1, 4, 16\}$ INT8/FP8 QK quantization) into PrimsTS block-sparse Video Sparse Attention (VSA) and Sparse Online Attention (SOL) on Blackwell (`SM100`/`SM103`), and why must the TMA descriptor clear bit 85 when the contiguous Q prefix is smaller than `128 KiB`?
2. **TensorRT-LLM INT8 Weight-Only (`W8A16`) FusedMoE 64-Row Tile Alignment Under Tensor Parallelism (`NVIDIA/TensorRT-LLM@64ef54ce1249`, `#16198`) and Model Express (`MX`) `Phi3ForCausalLM` Qualification (`@b78033a95183`, `#19997`)**: Why does `preprocess_weights_for_mixed_gemm` fail when `moe_intermediate_size // tp_size` (e.g., `1856 // 2 = 928` or `200`) is not a multiple of `64`, and why must the `fc1`/`fc2` intermediate dimension check run *after* `makeActivationParams`?
3. **vLLM `LoRAFusedMoEModularMethod` `_lora_context` Loss During `reload_weights` (`vllm-project/vllm@7dc44a00161a`, `#63319`) and ROCm MI355 AITER `W4A16` MXFP4 `SWIGLUOAI_UNINTERLEAVE` (`@be03d19750cc` `#62907`, `@f869d6478838` `#63283`)**: Why did reloading base weights on a LoRA-wrapped quantized MoE layer (`Qwen3-30B-A3B-quantized.w4a16`) silently drop `_lora_context` and retain stale expert weights?
4. **SGLang Rust Server Prefill Bootstrap Port Aliasing (`sgl-project/sglang@0d592c475bca`, `#46066`) and `/v1/responses` Muse Namespace Tools (`@6a319e7b3446`, `#44685`)**: Under `SGLANG_RUST_SERVER=1`, why did aliasing `disaggregation_bootstrap_port` to `--port` break transfer backends that run their own `BaseKVBootstrapServer`?

---

## 2. Core Implementation Paths & Source-Level Mechanics

### 2.1 TensorRT-LLM PrimsTS SageAttention for VSA / SOL & Blackwell TMA/LDTM Adaptation (`#20224`)

In `tensorrt_llm/_torch/visual_gen/jit_kernels/flash_attention/cute/` (`sage.py`, `sage_quant.py`, `block_sparse_attention.py`) and `cpp/tensorrt_llm/kernels/cutlass_kernels/primsts/`, TensorRT-LLM wires PrimsTS SageAttention into block-sparse VSA and SOL on `SM100`/`SM103`:

- **Flat Sentinel-Padded Scale Geometry (`sage_quantize`)**:
  - For a ragged batch of $B$ sequences with total tokens $T$ and quantization block size $b_{\text{blk}} \in \{1, 4, 16\}$, `q_scale` and `k_scale` use flat shape $[H, \lceil T / b_{\text{blk}} \rceil + B - 1]$, where sequence $b$ begins at slot:
    $$
    \text{slot}_b = \lfloor \text{cu\_seqlens}[b] / b_{\text{blk}} \rfloor + b
    $$
  - `v_scale`, `v_mean` (`smooth_v=True`), and `k_mean` (`smooth_k=True`) have shape $[B, H_{KV}, D]$, backed by a dedicated `int64_t` workspace slice (`sageKMeanSum`) for exact per-sequence K-mean accumulation.
- **Proxy Route Summary Reuse & `_PrepareSageScales` + PDL Overlap**:
  - When `use_proxy_routes=True`, proxy summaries (`q_summary`, `k_summary`) are quantized at $b_{\text{blk}}=1$ into `q_summary_scale` / `k_summary_scale` while reusing the main attention's `v_scale` and `k_mean`.
  - `_PrepareSageScales` packs scales into `float4` ahead of attention (`_SAGE_ATOM_TOKENS=64`, `_SAGE_ATOM_MIN_WORDS=2`), and route preparation launches with `use_pdl=True` to overlap with scale preparation.
- **Blackwell `LDTM.STAT` & TMA Bit-85 OOB Workaround**:
  - Enables `tcgen05.ld.red.sync.aligned.32x32b.x{num}.f32.max` (`LDTM.STAT`) on `SM103`/`SM107` for row-max reduction, and clears bit 85 (`word 1 &= ~(1u << 21)`) on the Q TMA descriptor whenever the contiguous Q prefix is under `128 KiB` (`< 32` rows at BF16 $H=16, D=128$).
- **Measured B200 Lossy Accuracy Bounds (`author-benchmark`)**:
  - **INT8 QK**: worst `relative_l2 = 0.03618`, `cosine = 0.999403`, `max_abs = 0.140625`;
  - **FP8 E4M3 QK**: worst `relative_l2 = 0.04979`, `cosine = 0.998761`, `max_abs = 0.167969`.

### 2.2 TensorRT-LLM INT8-woq MoE 64-Row Tile Padding (`#16198`) & Model Express Phi-3 Qualification (`#19997`)

1. **INT8 Weight-Only (`W8A16`) MoE 64-Row Padding (`NVIDIA/TensorRT-LLM@64ef54ce1249`, `#16198`)**:
   - CUTLASS mixed-GEMM weight preprocessing requires each partition's row count to be a multiple of `ALIGN = 128 * 8 // 16 = 64`. When `moe_intermediate_size = 1856` is sharded across `tp_size = 2` (`928` rows/rank, remainder `32`) or `moe_intermediate_size = 200` (`256` padded), `quantization.py` now pads `w3` and `w1` individually to `inter_padded = ceil(inter_size / 64) * 64` *before* concatenating `w3_w1` (`[E, H, 2 * inter_padded]`) and pads `w2`'s input dimension to `inter_padded` (`[E, inter_padded, H]`).
   - In `fused_moe_cute_dsl.py`, the `fc1`/`fc2` intermediate-size `TORCH_CHECK` is moved *after* `makeActivationParams` so gated activation promotion (`SwigluBias`, `expand_ratio=2`) is accounted for before checking `fc1.size(2) == fc2.size(1) * expand_ratio`.
2. **Model Express (`MX`) `Phi3ForCausalLM` Qualification (`@b78033a95183`, `#19997`)**:
   - Registers `phi3-for-causal-lm-bf16-target-v1` (ABI `trtllm-phi3-dense-target-layout-v1`) for post-transform weight transfer across `TP=1, 2` while rejecting LongRoPE, tied embeddings, or mismatched `max_seq_len`.

### 2.3 vLLM LoRA MoE `_lora_context` Rebinding (`#63319`) & ROCm MI355 MXFP4/FP8 Verification (`#62907`, `#63283`)

1. **`LoRAFusedMoEModularMethod.process_weights_after_loading` Fix (`vllm-project/vllm@7dc44a00161a`, `#63319`)**:
   - Previously, `FusedMoEModularMethod.process_weights_after_loading` did not delegate to `self.old_quant_method.process_weights_after_loading(layer)`. When `reload_weights` was called on a LoRA-wrapped quantized MoE layer, `old_quant_method` built a new `moe_kernel` without `_lora_context` while `self.moe_kernel.impl` retained stale weights.
   - Now, `LoRAFusedMoEModularMethod` delegates to `old_quant_method`, copies `_lora_context` onto `new_kernel.fused_experts` and `new_kernel.prepare_finalize` via `set_lora_context`, updates `self.moe_kernel.impl = new_kernel.impl`, and re-aliases `self.old_quant_method.moe_kernel = self.moe_kernel`.
2. **ROCm MI355 AITER `W4A16` MXFP4 `SWIGLUOAI_UNINTERLEAVE` (`@be03d19750cc`, `#62907`) & Online FP8 Batch-Invariance (`@f869d6478838`, `#63283`)**:
   - Validates MiniMax-M3 `SWIGLUOAI_UNINTERLEAVE` (`limit=7.0, alpha=1.702, beta=1.0`, forcing `interleave=True` in AITER Triton MXFP4 kernels) and MI300/MI355 online FP8 MoE bitwise batch invariance (`bs1 == bsN`).

### 2.4 SGLang Rust PD Bootstrap Port Preservation (`#46066`) & Muse Namespace Tools (`#44685`)

1. **Rust Server Prefill Bootstrap Port Guard (`sgl-project/sglang@0d592c475bca`, `#46066`)**:
   - Introduces `rust_server_serves_kv_bootstrap(transfer_backend)` (`issubclass(server_class, CommonKVBootstrapServer)`). Only built-in HTTP registry backends alias default `8998 -> api_port (31000)` and reject conflicting custom ports (`33000 -> ValueError`); transfer backends with their own `BaseKVBootstrapServer` preserve their configured `disaggregation_bootstrap_port` (`8998` or `33000`).
2. **Muse `/v1/responses` Namespace Tools & `defer_loading` (`@6a319e7b3446`, `#44685`)**:
   - Preserves nested `type="namespace"` tool hierarchies and `defer_loading: True` flags in `req._template_tools` for chat-template rendering while flattening tools to `<namespace>.<function>` (e.g., `browser.search`) for `FunctionCallParser`.

---

## 3. Deterministic Offline Verification (`own-offline`)

Experiment script: `experiments/trtllm-primsts-sage-vsa-sol-woq-align64-vllm-lora-sglang/run.py` (results sealed in `summary.json` and guarded by `tests/test_phase71_experiments.py`):

| Verification Target | Key Inputs / Parameters | Measured / Simulated Output | Architectural Conclusion |
|---|---|---|---|
| TRT-LLM PrimsTS SageAttention VSA/SOL | $B=2, S_Q=S_{KV}=1024, H_Q=16, H_{KV}=2, b_K=16$ | `q_scale=[16, 2049]`, `k_scale=[2, 129]`, `k_summary_scale=[2, 33]`, Seq1 K slot `65` | Validates flat sentinel scale layout & proxy reuse; B200 INT8 cosine `0.999403` |
| TRT-LLM INT8-woq MoE 64-Row Padding | `inter=1856, TP=2` (`928`/rank, gated) | `inter_padded=960` (`+32` rows), `w3_w1=[8, 2048, 1920]`, `w2=[8, 960, 2048]` | Eliminates non-64-aligned TP shard crashes and orders `fc1/fc2` check after promotion |
| vLLM LoRA MoE Reload & MI355 MXFP4 | `LoRAFusedMoEModularMethod.reload_weights` | Delegates to `old_quant_method` and rebinds `_lora_context` & `moe_kernel.impl` | Fixes stale weights and dropped LoRA context on quantized MoE weight reload |
| SGLang Rust PD Bootstrap Port Routing | `api_port=31000`, `bootstrap_port=8998 / 33000` | HTTP registry: `8998->31000`, `33000->ValueError`; custom backend: keeps `8998 / 33000` | Prevents Rust HTTP server from hijacking dedicated KV transfer bootstrap ports |

---

## 4. Solutions Architect Exercises & Falsifiable Questions

1. **Calculation Exercise**: For a ragged batch of $B=3$ sequences with `cu_seqlens = [0, 512, 1536, 2048]`, `num_kv_heads = 4`, and `k_block_size = 16` in TensorRT-LLM PrimsTS SageAttention, compute the shape of `k_scale` and the starting slot index for sequence `b=2`.
   - *Solution*: Total tokens $T = 2048$, flat length $= \lceil 2048 / 16 \rceil + 3 - 1 = 130 \implies \text{shape } [4, 130]$. Sequence `b=2` starts at $\lfloor 1536 / 16 \rfloor + 2 = 98$.
2. **Falsifiable Hypothesis**: In a gated INT8-woq MoE layer (`expand_ratio=2`), is appending `2 * pad_rows` zeros to the end of the already-concatenated `w3_w1` tensor mathematically equivalent to padding `w3` and `w1` individually before concatenation?
   - *Falsification*: No—gated SwiGLU/GeGLU splits `fc1` output at the midpoint `inter_padded`. Padding only after concatenation shifts `w1` rows into the first half (`w3`) and misaligns every elementwise gate-up product.
