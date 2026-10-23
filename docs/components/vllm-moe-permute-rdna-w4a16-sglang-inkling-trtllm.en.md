# Deep Dive into vLLM `moe_permute` Padding-Route Remapping Without `expert_map`, ROCm RDNA (`gfx1151`) W4A16 `v_and_or_b32` Packed Dequantization, SGLang Inkling BF16-In/FP32-Out LM-Head Fidelity, Quark Block-FP8 Loading, and TensorRT-LLM CBTS Dataclass Bounding

> **Language / 语言**: [English](vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm.en.md) | [中文](vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm.md)  
> **Phase**: 70 (`priority: 79`, Track B — LLM Serving, Quantization & MoE Execution)  
> **Evidence Tiers**: `implementation` (fixed-commit source audit) + `own-offline` (deterministic standard-library simulation `experiments/vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm/run.py`)

---

## 1. Architectural Context & Core Engineering Questions

Recent commits across vLLM, SGLang, and TensorRT-LLM resolve five subtle serving, quantization, and coverage-selection invariants spanning CUDA, ROCm CDNA3 (`gfx942`), and ROCm RDNA3.5 (`gfx1151`):

1. **vLLM `moe_permute` Invalid `topk_ids == -1` Sorting Shift & Dynamic-Quantization Workspace Leakage Without `expert_map` (`vllm-project/vllm@e4db46c79bca`, `#59717`)**: When padding tokens or masked routes set `topk_ids = -1` without an `expert_map`, why did `cub::DeviceRadixSort::SortPairs` sort `-1` **before** Expert 0 and shift every valid expert segment, and why did unzeroed tail rows in `mm1_out` corrupt the per-tensor dynamic activation scale in CUTLASS FP8 and FusedHummingMoE?
2. **vLLM ROCm RDNA (`gfx1151`) W4A16 Prefill Packed FP16 Dequantization (`vllm-project/vllm@96bbaa300fe4`, `#60644`)**: How does a single inline AMDGCN instruction `v_and_or_b32 $0, $1, $2, 0x64006400` unpack two INT4 nibbles into two `1024.0 + nibble` FP16 lanes so zero-point subtraction and scaling fuse into packed `v_pk_fma_f16`?
3. **vLLM ROCm `gfx942` Kimi-K3 Load-Time INT4 Expert Requantization Validation (`vllm-project/vllm@2c58be22b68d`, `#63131`) and Speculative Decoding Drafter LoRA Bypass (`@58301f6cb9b1`, `#63186`)**: Why did `AiterMxfp4Experts` fail closed on MI300X (`gfx942`) when loading Kimi-K3, and why did `LoRAConfig.verify_with_model_config` reject speculative decoding when re-validating against a draft model such as `DeepSeekMTPModel`?
4. **SGLang Inkling `enable_lm_head_output_fp32=True` Preserving BF16 Operands with FP32 Outputs (`sgl-project/sglang@855c092deb93`, `#45335`)**: When two vocabulary candidates have exact BF16 dot products $1 + 2^{-10}$ and $1 + 2^{-9}$, why does a BF16 output buffer round both to `1.0` and flip greedy `argmax` from `1` to `0`, and how does `torch.mm(..., out_dtype=torch.float32)` preserve the winner without doubling LM-head weight memory?
5. **SGLang AMD Quark Block-FP8 DeepSeek Loader Fixes (`sgl-project/sglang@9d7ad308bdbb`, `#40776`) & TensorRT-LLM CBTS Top-Level Dataclass Consumer Bounding (`NVIDIA/TensorRT-LLM@dbc875d517fb` `#20411`, `@461183a90a04` `#20883`)**: How does `_resolve_quark_fp8_block_scale_name` remap Quark's `weight_scale` to `weight_scale_inv` without breaking MXFP4 `w13_weight_scale`?

---

## 2. Core Implementation Paths & Source-Level Mechanics

### 2.1 vLLM `moe_permute` Invalid `topk_ids` Remapping & Dynamic Scale Bounding (`#59717`)

In `csrc/moe/moe_permute_unpermute_op.cu`, `csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.inl`, `cutlass_moe.py`, and `fused_humming_moe.py`, vLLM fixes three coupled padding-route bugs:

1. **Signed Radix-Sort Misordering Without `expert_map`**:
   - Previously, `remap_expert_ids` was launched only when `expert_map.has_value()` was true. Without `expert_map`, raw signed `int32` `topk_ids` were passed directly to `cub::DeviceRadixSort::SortPairs`, placing `-1` padding routes **ahead of Expert 0** while `compute_and_sum_expert_offsets` ignored negative IDs—shifting every expert's slice `[expert_offsets[e], expert_offsets[e+1])` into the wrong rows.
   - **Fix**: `remap_expert_ids` now runs unconditionally; when `expert_map == nullptr`, any route passing `uint32_t(id) < uint32_t(n_expert)` stays `id` and all invalid/negative IDs map to the positive sentinel `n_expert`, sorting them after all valid expert rows.
2. **CUTLASS FP8 Unwritten Workspace Scale Contamination (`cutlass_moe.py`)**:
   - Under `per_act_token=False`, the dynamic per-tensor scale of `act_out` reduces over every row of `mm1_out`. Because grouped GEMM never writes rows sorted to sentinel `n_expert`, stale data in those tail rows leaked into the scale unless `mm1_out.fill_(0)` ran. `mm1_out.fill_(0)` now executes whenever `not per_act_token`.
3. **FusedHummingMoE Dynamic Input Quantization Bound (`fused_humming_moe.py`)**:
   - `num_valid_tokens = expert_offsets[-1:]` is now passed unconditionally so dynamic input quantization never reads past the valid routed rows.

### 2.2 vLLM ROCm RDNA (`gfx1151`) W4A16 Packed FP16 Dequantization (`#60644`)

In `vllm/model_executor/layers/quantization/kernels/mixed_precision/rdna_hybrid_w4a16.py`, the Triton W4A16 prefill kernel introduces `_int4_pair_to_fp16x2` for RDNA3.5 (`gfx1151`):

- **IEEE-754 FP16 Bias Trick (`0x64006400`)**:
  - In FP16, `0x6400` represents $1024.0$ (biased exponent $25 = 15 + 10$, mantissa $0$), where $1\text{ ULP} = 2^{10-10} = 1.0$.
  - Thus, for any 4-bit unsigned nibble $q \in [0, 15]$, the bit pattern `0x6400 | q` evaluates exactly to $1024.0 + q$.
  - One inline instruction `v_and_or_b32 $0, $1, $2, 0x64006400` masks two nibbles (`0x000F000F`) in low and high 16-bit halves and ORs `0x64006400`, producing two FP16 values $(1024.0 + q_{\text{lo}}, 1024.0 + q_{\text{hi}})$ in a single cycle.
- **Folding `1024.0` into the Zero-Point Subtrahend**:
  $$
  w = b_{\text{raw}} \cdot s - (1024.0 + z) \cdot s
  $$
  With symmetric $z = 8$ ($\text{sub} = 1032.0 \cdot s$) or asymmetric $z = \text{zp}$ ($\text{sub} = (1024.0 + \text{zp}) \cdot s$), interleaving adjacent lanes allows the compiler to emit packed `v_pk_fma_f16` instructions with `gfx1151`-tuned tiles (`BLOCK_N=64, BLOCK_K=64`).

### 2.3 ROCm `gfx942` Kimi-K3 INT4 Expert Requant Validation (`#63131`) & Drafter LoRA Bypass (`#63186`)

1. **`gfx942` Kimi-K3 INT4 Requantization (`#63131`)**: `AiterMxfp4Experts._supports_quant_scheme` and `make_mxfp4_moe_quant_config` now accept `(kInt4Static32, None)` when `on_gfx942()` is true, matching Kimi-K3's load-time requantization of MXFP4 experts to packed INT4 on MI300X while keeping native `(kMxfp4Static, *)` restricted to `gfx950`.
2. **Speculative Decoding Drafter LoRA Validation (`#63186`)**: `LoRAConfig.verify_with_model_config` now skips the `not model_config.is_lora_supported` error when `model_config.runner_type == "draft"`, allowing speculative decoding drafters (`DeepSeekMTPModel`) to coexist with target-model LoRA adapters.

### 2.4 SGLang Inkling `enable_lm_head_output_fp32=True` Preventing Greedy Flips (`#45335`)

In `python/sglang/srt/configs/inkling.py` and `python/sglang/srt/layers/logits_processor.py`:

- **Why BF16 Output Rounding Flips Greedy `argmax`**:
  - With `hidden = [1.0, 1.0]` and `head.weight = [[1.0, 1/1024], [1.0, 1/512]]` (all exact in BF16), the exact dot products are $1 + 2^{-10} = 1.0009765625$ and $1 + 2^{-9} = 1.001953125$ (`argmax = 1`).
  - Because BF16 has a 7-bit mantissa ($1\text{ ULP} = 2^{-7} = 1/128$ on $[1, 2)$), storing the output in BF16 rounds both scores to `1.0`, tying them so `argmax` returns `0`.
- **Zero-Weight-Expansion Solution**:
  - Instead of `enable_fp32_lm_head` (which promotes the entire LM-head weight matrix to FP32), `enable_lm_head_output_fp32=True` keeps `hidden_states` and `lm_head.weight` in BF16 and invokes `torch.mm(hidden_states, lm_head.weight.T, out_dtype=torch.float32)` on CUDA (`use_mm_out_dtype`), preserving exact FP32 logits and `argmax == 1`.

### 2.5 SGLang Quark Block-FP8 DeepSeek Loader (`#40776`) & TensorRT-LLM CBTS Dataclass Bounding (`#20411`)

1. **SGLang Quark Block-FP8 DeepSeek Loader (`#40776`)**:
   - `_resolve_quark_fp8_block_scale_name(name, params_dict)` maps `*.weight_scale` to `*.weight_scale_inv` only when `name` is absent from `params_dict` and the `_inv` target exists, fixing Quark QKV/indexer block-FP8 scales while preserving MXFP4 `w13_weight_scale`.
   - `_resolve_layer_weight_block_size(layer, quant_config)` checks `layer.quant_method.weight_block_size` (`[64, 128]`) before falling back to `quant_config.linear_fp8_config.weight_block_size` (`[128, 128]`).
2. **TensorRT-LLM CBTS Top-Level Dataclass Bounding (`#20411`) & Forward-Error Survival (`#20883`)**:
   - CBTS coverage selection now bounds newly added top-level `@dataclass` declarations (with optional literal `frozen` bool, no base classes or metaclasses, and unshadowed builtin annotations) through the functions consuming the class binding, and updates `test_forward_error_survival.py` fakes with `kv_cache_retention_config=None` and `_is_kv_manager_v2=False`.

---

## 3. Deterministic Offline Verification (`own-offline`)

Experiment script: `experiments/vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm/run.py` (results saved in `summary.json`):

| Verification Target | Parameters | Measured / Simulated Output | Architectural Takeaway |
|---|---|---|---|
| vLLM `moe_permute` invalid IDs without `expert_map` | 3 valid + 2 pad tokens (`topk=4, E=8`) | Legacy first key `-1` (shifted offsets); fixed first key `0`, 8 tail sentinels `8`; `51.2x` scale inflation prevented | Invalid `topk_ids` must be remapped to `n_expert` and excluded from dynamic scale reductions |
| ROCm RDNA (`gfx1151`) W4A16 `v_and_or_b32` | Sym `(3, 14), s=0.0625`; Asym `(5, 11), s=0.125, zp=6` | Packed words `0x640E6403` / `0x640B6405`, max abs error `0.0` | Single-instruction `0x64006400` unpack + folded zero-point is bit-exact |
| SGLang Inkling `enable_lm_head_output_fp32` | `w=[[1, 1/1024], [1, 1/512]]`, `h=[1, 1]` | BF16 output `[1.0, 1.0]` (`argmax=0`) vs FP32 output `[1.0009766, 1.0019531]` (`argmax=1`) | `out_dtype=torch.float32` prevents greedy flips at `0.5x` the weight memory of an FP32 head |
| SGLang Quark Block-FP8 Loader | `qkv_a.weight_scale` + `experts.w13_weight_scale`, layer `[64, 128]` | Maps QKV to `weight_scale_inv`, keeps `w13_weight_scale`, resolves `[64, 128]` | Safely supports mixed Quark FP8 attention + MXFP4 expert checkpoints |

---

## 4. Solutions Architect Exercises & Falsifiable Questions

1. **Bit-Level Exercise**: In `_int4_pair_to_fp16x2`, why is the constant `0x6400` ($1024.0$) chosen rather than `0x3C00` ($1.0$)?
   - *Answer*: FP16 has 10 mantissa bits, so at `0x3C00` ($1.0$) bit 0 has weight $2^{-10}$. Adding $10$ to the biased exponent ($15 + 10 = 25 \implies \text{0x6400} = 1024.0$) makes bit 0 have weight $2^0 = 1.0$, so OR-ing a 4-bit integer $q$ directly yields $1024.0 + q$.
2. **Falsifiable Hypothesis**: Does setting `enable_fp32_lm_head=True` provide the same memory efficiency as `enable_lm_head_output_fp32=True` for a 128K-vocabulary model with `H=4096`?
   - *Falsification*: `enable_fp32_lm_head=True` promotes the $[131072, 4096]$ weight matrix from BF16 (`1.0 GiB`) to FP32 (`2.0 GiB`), costing an extra `1.0 GiB` of HBM per TP rank, whereas `enable_lm_head_output_fp32=True` keeps the `1.0 GiB` BF16 weights and only writes the per-batch logits buffer in FP32.
