# SGLang CPU Offloader Storage-View Preservation, TensorRT-LLM Packed 5D QKV Ulysses Unscatter & vLLM Rust Bad-Words Parity

> **Language**: [中文](sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words.md) | [English](sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words.en.md)
> **Phase**: 65 (`priority: 74`) · **Track B**: Inference Serving, Diffusion & Multi-Node Sequence Parallelism
> **Verified Upstream Sources**: `sgl-project/sglang@6d6c9cd4ae` (`#44139`), `sgl-project/sglang@494eeecb6c` (`#45944`), `sgl-project/sglang@f9129915a3` (`#45994`), `NVIDIA/TensorRT-LLM@1586d85375` (`#17147`), `NVIDIA/TensorRT-LLM@8d1c7e0b26` (`#20660`), `vllm-project/vllm@34b2954482` (`#63315`)

---

## 1. Core Architectural Problem & Background

Across hybrid linear attention models (KDA / `RadixLinearAttention`), long-sequence multimodal diffusion pipelines (Qwen-Image / Wan TI2V with Ulysses Sequence Parallelism), and high-throughput Rust serving frontends, subtle mismatches in **tensor storage aliasing**, **TorchDynamo dict-size guards**, **post-All-to-All memory unscattering**, and **cross-language tokenizer validation** directly govern system correctness and tail latency:

1. **Stale Weight Views Under SGLang CPU Offload (`sglang@6d6c9cd4ae`)**: `torch.func.functional_call` only swaps registered `nn.Parameter`s and buffers. However, several SGLang layers cache storage-sharing views or derived tensors as plain Python attributes (e.g., KDA's `attn.conv_weights` viewing `qkv_conv1d.weight`, or `GemmaRMSNorm.gemma_weight` storing `weight + 1.0`). Offloading `qkv_conv1d.weight` to CPU left `attn.conv_weights` pointing to stale storage.
2. **TorchDynamo Recompile Storms from Diffusion Verdict Caching (`sglang@494eeecb6c` & `f9129915a3`)**: `can_use_linear_gelu(linear, x)` memoized module eligibility in a global `_VERDICTS` dict. Reading and mutating `_VERDICTS` inside `torch.compile` caused TorchDynamo to guard on `_VERDICTS` length, triggering a full graph recompilation on every new linear layer. Separately, Wan TI2V failed to broadcast sampled timesteps across Sequence Parallel (SP) ranks (`f9129915a3`).
3. **Intermediate HBM Traffic in TensorRT-LLM Ulysses Post-All-to-All Unscatter (`TensorRT-LLM@1586d85375`)**: In Ulysses sequence parallelism for Qwen-Image, the head-to-sequence All-to-All produces a packed 5D receive buffer `[P, B, Sp, 3, H, D]`. Previously, Python executed `permute -> contiguous -> unbind` before calling `launchUlyssesPostUnscatter`, doubling HBM read/write traffic (4× tensor bytes) and adding extra kernel launches.
4. **vLLM Rust Frontend `bad_words` Semantic Drift (`vllm@34b2954482`)**: vLLM's Rust frontend lacked pre-tokenization empty-string rejection, duplicate filtering, `max_num_bad_words` enforcement, and exact `trim_start()` + prefix-space deduplication parity with Python's `sampling_params.py`.

---

## 2. Source-Level Mechanics & Mathematical Formulation

### 2.1 SGLang Generic `untyped_storage().data_ptr()` Scanning & Alias Refresh (`6d6c9cd4ae` & `494eeecb6c`)

In `python/sglang/srt/utils/offloader.py`, `_get_resident_parameter_ids(module)` replaces fragile layer-specific checks with a generic storage-pointer sweep:
1. Scan `vars(owner).values()` across all submodules and record `(value.device, value.untyped_storage().data_ptr())` in `cached_storages` for every non-`Parameter`, non-meta tensor attribute with `numel() > 0`.
2. Pin any `nn.Parameter` whose `(parameter.device, parameter.untyped_storage().data_ptr())` appears in `cached_storages` as GPU-resident.
3. In `loader.py`, `_rebind_parameter_aliases` invokes `GemmaRMSNorm.refresh_gemma_weight()` and `RadixLinearAttention.refresh_conv_weights(cweight)`.
4. In `fused_linear_gelu_site.py` (`494eeecb6c`), `if torch.compiler.is_compiling(): return _input_independent_ok(linear)` bypasses `_VERDICTS` during tracing, collapsing $L$ recompilations into `1`.

### 2.2 TensorRT-LLM `ulyssesPackedQkvPostUnscatterKernel` & Int32 Overflow Guards (`1586d85375`)

In `cpp/tensorrt_llm/kernels/ulyssesPostUnscatterKernel.cu`, a dedicated CUDA kernel consumes `[P, B, Sp, 3, H, D]` directly:
- **Grid & Thread Mapping**: Each thread copies 16 bytes (`uint4` = 8 `BF16` elements, `VEC = 8`). With $\text{vec\_per\_row} = D / 8$:
  $$\text{blockDim.x} = H \times \frac{D}{8} \le 1024, \quad \text{gridDim} = (3 \cdot P \cdot S_p,\ B,\ 1)$$
- **Direct 6D-to-4D Index Mapping**: With $\text{PSp} = P \cdot S_p$, $qkv = \lfloor \text{blockIdx.x} / \text{PSp} \rfloor \in \{0, 1, 2\}$, $psp = \text{blockIdx.x} \bmod \text{PSp}$, $p = \lfloor psp / S_p \rfloor$, and $sp = psp \bmod S_p$:
  $$\text{in\_base} = \Big(\big(((p \cdot B + b) \cdot S_p + sp) \cdot 3 + qkv\big) \cdot H + h\Big) \cdot D + \text{vec\_idx} \cdot 8$$
  $$\text{out\_base} = \Big((b \cdot \text{PSp} + psp) \cdot H + h\Big) \cdot D + \text{vec\_idx} \cdot 8$$
  routing `qkv == 0/1/2` directly to `q_out`, `k_out`, or `v_out` and cutting post-All-to-All HBM traffic by **50.0%**.
- **Int32 Overflow Guards (`ulyssesPostUnscatterOp.cpp`)**: `checkInt32Dim`, `checkInt32Product`, and `checkInt32SumProduct("P*(Sp_q+Sp_k+Sp_v)", ...)` prevent 32-bit launch grid truncation on large sequences.

### 2.3 vLLM Rust `tokenize_bad_words` Parity (`34b2954482`)

`rust/src/text/src/lower.rs` enforces five exact invariants matching Python `sampling_params.py`:
1. Reject any empty string (`""`) with `TokenIdsError::EmptyBadWord` before tokenizing.
2. Deduplicate input strings via `HashSet` in $\mathcal{O}(1)$ before tokenization and enforce `max_num_bad_words()`.
3. Strip leading whitespace (`trim_start()`), encode both `without_space` and `with_space`, and keep `with_space` if and only if:
   $$T_{\text{sp}}[0] \neq T_{\text{no\_sp}}[0] \ \land\ |T_{\text{sp}}| == |T_{\text{no\_sp}}|$$

---

## 3. Deterministic Verification (`experiments/sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words`)

Running `python experiments/sglang-offloader-trtllm-ulysses-qkv-vllm-bad-words/run.py --check` verifies all 7 invariants:
- **SGLang Offloader & Compile**: Automatically pins `kda.qkv_conv1d.weight` (`storage_ptr=2002`) on GPU while offloading unaliased MLP weights (`storage_ptr=1001`), and reduces 40-layer Dynamo compilations from `40` to `1`.
- **TRT-LLM Packed QKV Ulysses**: At `P=2, B=1, Sp=4096, H=24, D=128`, uses `384` threads/block (`<= 1024`) and cuts HBM traffic from `301,989,888 B` to `150,994,944 B` (**50.0% reduction**).
- **vLLM Rust Bad-Words**: Rejects empty words and >128 lists immediately, and produces exact token sequences `[[101], [201], [102, 103], [104, 105]]`.
