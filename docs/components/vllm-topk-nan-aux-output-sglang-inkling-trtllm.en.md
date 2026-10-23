# Deep Dive into vLLM Cooperative Top-K NaN Inline-PTX Ordering Fix, AuxOutput KV-Tier Retention Alignment, DeepSeek-V4.1-Flash PCP SWA Bounded Replay, SGLang Inkling TMLv0 Parsing, and TensorRT-LLM Native Payload Migration

> **Language / 语言**: [English](vllm-topk-nan-aux-output-sglang-inkling-trtllm.en.md) | [中文](vllm-topk-nan-aux-output-sglang-inkling-trtllm.md)  
> **Phase**: 69 (`priority: 78`, Track B — Serving, Scheduling & Speculative Decoding)  
> **Evidence Tiers**: `implementation` (fixed-commit source audit) + `own-offline` (deterministic standard-library simulation `experiments/vllm-topk-nan-aux-output-sglang-inkling-trtllm/run.py`)

---

## 1. Architectural Context & Core Engineering Questions

Across multi-rank Prefill Context Parallelism (`PCP`), routed-expert auxiliary output caching, high-throughput CUDA Top-K sampling, and multimodal/tool-calling serving engines, upstream commits in this cycle resolve five critical production correctness and memory/serialization bottlenecks:

1. **vLLM `cooperative_topk` / `persistent_topk` / `sampled_topk` Illegal Memory Access on `NaN` Scores (`vllm-project/vllm@109669cfb314`, `#63118`)**: Why does `nvcc` lower the integer bit-twiddling expression `uint32_t bits = __float_as_uint(x); return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);` into the floating-point instruction `-|x|`, canonicalizing quiet `NaN` (`0x7fc00000`) to the ordered key of `-0.0` (`0x7fffffff`) and triggering an out-of-bounds memory write when `NaN` counts exceed `top_k`?
2. **vLLM `AuxOutput` (e.g., MoE `routed_experts`) Retention Alignment Across Local/External KV Cache Tiers (`vllm-project/vllm@aea0513a322e`, `#62591`) and MRV2 Profiling Cache Exclusion (`@fbe133ff30d3`, `#61509`)**: When KV blocks are offloaded to an external KV connector or partially evicted under a sliding window, how does the scheduler prevent prefix-cache hits from outrunning retained `routed_experts` slots?
3. **vLLM DeepSeek-V4.1-Flash Decoder-Side SWA Bounded Replay Under `PCP > 1` (`vllm-project/vllm@c7476c7a6fc8`, `#62384`)**: When Prefill Context Parallelism interleaves tokens across ranks, different ranks retain unequal numbers of sliding-window rows (`num_kept`). How are collective tensor shapes kept uniform across ranks?
4. **SGLang Inkling `tml-renderers` Token-ID State-Machine Parser (`sgl-project/sglang@8b9f72417fa8`, `#45340`) and Mistral3 `LlavaForConditionalGeneration` Subclassing (`@4542fc8f598e`, `#45656`)**: Why must control tokens like `<|end_message|>` be matched by token ID rather than decoded strings?
5. **TensorRT-LLM `LlmRequest` / `executor::Result` Heavy-Tensor Migration to Python (`NVIDIA/TensorRT-LLM@e139d884a920`, `#20388`) and Multimodal Processor Chat-Template Fallback (`@f035671959d2`, `#20880`)**: How does moving `contextLogits`, `encoderInputFeatures`, and `additionalOutputs` out of C++ pybind serialization shrink IPC overhead?

---

## 2. Core Implementation Paths & Source-Level Mechanics

### 2.1 vLLM Top-K `NaN` Compiler Canonicalization Trap & Inline-PTX `mov.b32` Fix (`#63118`)

In `csrc/topk.cu` and `csrc/sampler_topk_kernels.cu`, vLLM's two-pass radix-bucket Top-K kernels suffered from a subtle compiler algebraic rewrite:
- **Mismatch Between Pass 1 (FP16 Coarse Binning) and Pass 2 (FP32 Refinement)**:
  - Pass 1 converts `float` scores to FP16 and maps the top 11 bits into 2048 coarse bins. Because `NaN` has all exponent bits set, it lands in the highest coarse bin (`bin 2047`), above `+inf` and all finite values.
  - Pass 2 calls `convert_to_uint32_v2(float x)` to map `float` values into monotonically ordered `uint32_t` keys. However, **`nvcc` recognized `bits | 0x80000000u` on `__float_as_uint(x)` as setting the sign bit and lowered it to the floating-point instruction `-|x|`**! On the GPU FP pipeline, `-|x|` canonicalizes quiet `NaN` (`0x7fc00000`), causing `~bits` to evaluate to `0x7fffffff`—the exact key of `-0.0`.
  - **Crash Mechanism**: Whenever an input row contained more than `top_k` `NaN` values, coarse bin `2047` overflowed into the refinement pass looking for keys near `0xffc00000`, but Pass 2 mapped every `NaN` to `0x7fffffff`, causing the refinement scan to miss them and write out of bounds.
- **Opaque Inline PTX Fix**:
  ```cuda
  uint32_t bits;
  asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(x));
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  ```
  Reading the raw bits through an opaque inline PTX `mov.b32` prevents `nvcc` from folding the bit manipulation back into an FP instruction, preserving strict monotonic order:
  $$
  -\infty\;(\texttt{0x007FFFFF}) < -0.0\;(\texttt{0x7FFFFFFF}) < +0.0\;(\texttt{0x80000000}) < +\infty\;(\texttt{0xFF800000}) < \text{NaN}\;(\texttt{0xFFC00000})
  $$

### 2.2 vLLM `AuxOutput` Retention Alignment Across KV Tiers (`#62591`) & MRV2 Profiling Cache Exclusion (`#61509`)

1. **`AuxOutputSchedulerConnector` Cross-Tier Retention Guard (`#62591`)**:
   - When `return_routed_experts=True` is active, each full KV block stores its routed-expert indices in a shared-memory `mmap` `BlockObjectStore` (sized to `num_blocks * 2` when an external KV connector is configured).
   - The scheduler tracks `retained_block_hashes` and `released_block_hashes`: during `get_computed_blocks`, `max_cache_hit(request)` truncates local prefix hits to the longest contiguous prefix of retained auxiliary blocks; for external KV connectors, `cap_external_hit(request, num_local_hit_tokens, num_external_hit_tokens)` rejects mid-block hits (e.g., `31` tokens) or unretained external blocks (dropping external hits to `0` so tokens are recomputed), verified across a 9-case scheduler matrix.
   - **Dynamic Arena Growth**: If `BlockObjectStore._evict_until_fits` finds every slot referenced by active requests, it logs a warning and grows the `mmap` arena to $\max(\text{nbytes},\; 2 \times \text{len}(\text{arena}))$ instead of raising `MemoryError`.
2. **MRV2 `adaptive_verification` Profiling Cache Cleanup (`#61509`)**:
   - In `vllm/v1/worker/gpu/model_runner.py` (`capture_model`), `speculator.run_draft_cost_profiling()` runs eager dummy steps before CUDA-graph capture, leaving `~2 GiB` of cached allocations in the PyTorch allocator. Calling `torch.accelerator.empty_cache()` immediately afterward prevents those reclaimable blocks from inflating `cuda_graph_memory_bytes`.

### 2.3 vLLM DeepSeek-V4.1-Flash Decoder-Side SWA Bounded Replay Under `PCP` (`#62384`)

In `vllm/model_executor/models/deepseek_v4_1_flash.py` and `vllm/v1/worker/gpu/pcp_utils.py`, vLLM enables SWA bounded replay under `PCP > 1`:
- **Minimum Window Invariant**: `init_spec_state` enforces `window >= 2 * pcp_size` (`window >= 4` for `pcp_size=2`) so every trimmed request has at least one token on every PCP rank.
- **Cross-Rank Tail Ownership & Zero-Padding (`replay_batch.num_padded`)**: Under `pcp_size = 2` and SWA `window = 128`, consider a batch with 1 single-token decode request and 1 513-token prefill resuming from a 1024-token APC hit (1537 total tokens). After PCP zigzag partitioning, Rank 0 holds 256 local tokens (`127` kept rows in the last 128 global positions: `[0]` and `[130..255]`), while Rank 1 holds 258 local tokens (`2` kept rows: `[256, 257]`). `DecoderReplayLayers._run` pads `row_inputs` on both ranks to `replay_batch.num_padded = 129` rows (padding `127` zero rows on Rank 1) before running the replay layers and slicing back to `num_kept`.

### 2.4 SGLang Inkling TMLv0 Parser (`#45340`), Mistral3 Inheritance (`#45656`) & TensorRT-LLM Native Payload Migration (`#20388`, `#20880`)

- **SGLang Inkling Token-ID State Machine (`#45340`)**: Integrates `tml-renderers==0.1.0` (`o200k_base_chat`). `InklingOutputParser.parse_delta_tokens` transitions strictly on special token IDs (`>= 200001`), so ordinary text tokens spelling `"<|end_message|>"` remain literal text; `reasoning_effort` maps presets (`none=0.0, minimal=0.1, low=0.2, medium=0.7, high=0.9, xhigh=max=0.99`) into $[0.0, 0.99]$ while rejecting `1.0` and booleans.
- **SGLang Mistral3 Subclassing (`#45656`)**: `Mistral3ForConditionalGeneration` directly subclasses `LlavaForConditionalGeneration` (`nn.Module`) so `model.eval()` returns the outer model rather than `self.inner`.
- **TensorRT-LLM `LlmRequest` / `Result` Migration (`#20388`) & Chat-Template Fallback (`#20880`)**: Moves heavy tensors (`contextLogits`, `generationLogits`, `encoderOutput`, `encoderInputFeatures`, `additionalOutputs`) out of C++ pybind serialization into Python, shrinking the C++ `executor::Result` pickle state tuple from **14 fields to 11 fields**; `resolve_hf_chat_template` checks `getattr(processor, "chat_template", None) is not None` before falling back to `tokenizer.get_chat_template(...)`.

---

## 3. Deterministic Offline Verification (`own-offline`)

Running `python experiments/vllm-topk-nan-aux-output-sglang-inkling-trtllm/run.py --check` produces `summary.json` and verifies all invariants:

| Verification Dimension | Metric / Scenario | Baseline / Negative Control | Optimized / Guarded Result | Evidence Tier |
|---|---|---|---|---|
| **vLLM Top-K `NaN` Ordering (`#63118`)** | `qNaN (0x7FC00000)` mapped `uint32` key | `nvcc` `-|x|` folds `NaN` to `0x7FFFFFFF` (collides with `-0.0`, OOB write) | Inline PTX `mov.b32` preserves `0xFFC00000 > +inf (0xFF800000)` | `implementation` + `own-offline` |
| **vLLM `AuxOutput` Retention (`#62591`)** | 9-case local/external/SW hit matrix & full arena | External hits outrun evicted aux blocks; full arena raises `MemoryError` | Hits match `[64, 32, 32, 32, 0, 0, 32, 0, 0]`; arena doubles (`6B -> 12B`, `4` slots) | `implementation` + `own-offline` |
| **vLLM PCP SWA Replay (`#62384`)** | `pcp_size=2, window=128` Rank 0 vs Rank 1 kept rows | Rank 0 keeps `127` rows, Rank 1 keeps `2` rows -> collective shape mismatch | Pads both ranks to `num_padded=129` (`127` zero rows on Rank 1); enforces `window >= 4` | `implementation` + `own-offline` |
| **SGLang Inkling & TRT-LLM (`#45340/#20388`)** | Literal `<|end_message|>`, `effort=1.0`; C++ `Result` pickle | String parser truncates literal text; C++ `Result` serializes 14 fields | Token-ID parser preserves literal text & rejects `1.0`; C++ `Result` tuple shrunk to **11 fields** | `implementation` + `own-offline` |

---

## 4. Exercises & Falsifiable Follow-Up Questions for Solution Architects

1. **Compiler Optimization Exercise (IEEE-754 Bit-Twiddling vs PTX `mov.b32`)**: Explain why `nvcc` converts `__float_as_uint(x) | 0x80000000u` into a floating-point negate-absolute-value instruction, and why hardware FP instructions canonicalize quiet `NaN` payloads before integer bitwise complement `~bits` executes.
2. **Falsifiable Hypothesis**: Under `return_routed_experts=True` with an external KV connector, if an external lookup returns a non-block-aligned token count (e.g., `31` tokens at `block_size = 16`), `AuxOutputSchedulerConnector.cap_external_hit` must clamp the external hit to `0` and force recomputation to avoid reading unpopulated routed-expert slots.
