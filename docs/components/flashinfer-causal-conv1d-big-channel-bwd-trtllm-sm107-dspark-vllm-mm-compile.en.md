# FlashInfer Big-Channel Causal Conv1D Backward Selection, TensorRT-LLM SM107 DSpark RMSNorm/RoPE Gate Decoupling & vLLM PaddleOCR-VL Vision Compile

> **中文版**: [flashinfer-causal-conv1d-big-channel-bwd-trtllm-sm107-dspark-vllm-mm-compile.md](flashinfer-causal-conv1d-big-channel-bwd-trtllm-sm107-dspark-vllm-mm-compile.md)  
> **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`  
> **Experiment**: [`experiments/flashinfer_causal_conv1d_bwd_trtllm_sm107_vllm_compile/run.py`](../../experiments/flashinfer_causal_conv1d_bwd_trtllm_sm107_vllm_compile/run.py)

---

## 1. Engineering Problem & Context

Hybrid attention/SSM and multimodal model stacks must route shapes to architecture-optimal kernels across `sm_100a` (B200), `sm_103a` (GB300), and `sm_107a` (R200) without sacrificing CUDA Graph capturability or bitwise reproducibility:
1. **FlashInfer `cake_causal_conv1d_train` Big-Channel Backward Latency (`flashinfer-ai/flashinfer@5f8f8aad`, `#6874`)**: The packed width-4 depthwise causal Conv1D fused with SiLU (BF16 I/O, FP32 math and deterministic `dweight`) needed a higher-occupancy 2-CTA/SM, 8-token staged TMA2D backward program for wide channel counts (`C >= 512` on `sm_100a`/`sm_103a`, `C >= 1024` on `sm_107a`).
2. **TensorRT-LLM SM107 DSpark Fused RMSNorm+RoPE Gate Coupling (`NVIDIA/TensorRT-LLM@083447cd`, `#20366`)**: `is_fused_dspark_rmsnorm_rope_supported` previously checked `_get_dspark_arch_str()`, which restricted plain SIMT CuTe RMSNorm+RoPE kernels to `SM100`/`SM103` alongside the narrower DSV4 DSpark attention gate.
3. **vLLM `PaddleOCR-VL` Vision Encoder Compilation (`vllm-project/vllm@0ca531ba`, `#61838`)**: `PaddleOCR-VL` compiled only its 2 Ernie-4.5 language model submodules while leaving all 27 `SiglipEncoderLayer` blocks in eager mode.

---

## 2. First-Hand Code Analysis & Quantitative Results

### 2.1 FlashInfer `channels_at_least` Routing and Capacity-Owner Workspace Sizing
In `flashinfer/cake_causal_conv1d_train_backend/cake_jit.py`, commit `5f8f8aad` registers `cake_causal_conv1d_train_90351ecca30604957251` (`vec=8, copy="tma2d", stage_tokens=8, min_blocks=2`, `dynamic_smem_bytes=82944`):
- **Host-Side Pure Rule**: `select_rule = {"kind": "channels_at_least", "channels": {"sm_100a": 512, "sm_103a": 512, "sm_107a": 1024}}`. `selected_modules(arch, "bwd", C=C)` depends only on `(arch, C)`—never on `S` or tensor contents—preserving bitwise-deterministic `dweight` reduction.
- **Workspace Capacity Owner (`workspace_record`)**: Each cooperative backward program writes FP32 `dweight` partials of shape `[rows, C, 4]` where `n_ch_tiles = ceil(C / (128 * 8))` and `rows = ceil(ctas_per_sm * SM_count / n_ch_tiles)`. Public allocation sizes the buffer by `max(ctas_per_sm)` across all backward programs on `arch` (`2` CTAs/SM), so smaller-grid fallback launches simply write and reduce their leading rows (`160` rows vs `320` capacity-owner rows at `C=256` on `sm_103a`) without zeroing or reallocation.
- **Author-Benchmarked Speedups (`author-benchmark`, CUDA 13.4)**:
  - **GB300 (`sm_103a`)**: `1.0569x` geomean across all 14 shapes (`1.0667x` across 12 routed shapes `C >= 512`), reaching `1.0991x` at `S=250,340, C=8192` (`2264.3 -> 2060.1 us`).
  - **B200 (`sm_100a`)**: `1.0765x` geomean overall (`1.0898x` routed), peaking at `1.1226x` at `C=1024` (`330.6 -> 294.5 us`).
  - **R200 (`sm_107a`)**: `1.0511x` geomean overall (`1.1908x` across 4 routed shapes `C >= 1024`), reaching `1.2581x` at `C=8192` (`1652.2 -> 1313.3 us`).

### 2.2 TensorRT-LLM SM107 DSpark Gate & vLLM `SiglipEncoderLayer` Compile
- **TRT-LLM (`083447cd`)**: Defines `_DSPARK_RMSNORM_ROPE_SM_VERSIONS = frozenset((100, 103, 107))` via `_is_dspark_rmsnorm_rope_sm_supported()`, enabling fused RMSNorm+RoPE on `SM107` while keeping `_get_dspark_arch_str(107) is None` for DSV4 DSpark attention.
- **vLLM (`0ca531ba`)**: Decorates `SiglipEncoderLayer` with `@support_torch_compile(dynamic_arg_dims={"hidden_states": 1, "cu_seqlens": 0, "rotary_pos_emb": 0}, enable_if=should_torch_compile_mm_encoder, is_encoder=True)`, increasing compiled modules from `2` (`compile_mm_encoder=False`) to `29` (`compile_mm_encoder=True`).

---

## 3. Architect Exercises & Falsifiable Questions

1. **Calculation**: Compute `n_ch_tiles`, `rows`, and `Bytes_bwd = 6SC + 32C + 32C * rows` on `sm_103a` (160 SMs) for `S=250,340, C=8192`.
   - **Answer**: `n_ch_tiles = 8`, `rows = ceil(2 * 160 / 8) = 40`, and `Bytes_bwd = 12,316,459,008 B` (`~12.316 GB`).
2. **Falsifiable Hypothesis**: Why must `cake_jit.specs_for_arch(arch)` build both the fallback and big-channel libraries ahead of CUDA Graph capture? Because a shape change from `C=256` to `C=512` selects a different compiled library and would otherwise trigger host JIT compilation inside stream capture.
