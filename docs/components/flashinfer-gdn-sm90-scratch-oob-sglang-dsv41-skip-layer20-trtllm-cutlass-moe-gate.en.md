# FlashInfer SM90 GDN Prefill Scratch OOB Fix, SGLang AMD DSV4.1 Layer-20 Bounded-Replay Skip, and TensorRT-LLM CUTLASS MoE / MLA Gates

> **中文版**: [flashinfer-gdn-sm90-scratch-oob-sglang-dsv41-skip-layer20-trtllm-cutlass-moe-gate.md](flashinfer-gdn-sm90-scratch-oob-sglang-dsv41-skip-layer20-trtllm-cutlass-moe-gate.md)

## 1. First-Hand Source Evidence

| System / Repository | Commit / Date | Core Source Paths | Evidence Class |
|---|---|---|---|
| **FlashInfer** (`flashinfer-ai/flashinfer`) | `afbe429cc26216343f00c17fc9aa23fa5d8449ed` (`2026-10-20`, PR `#6849`) | `flashinfer/gdn_prefill.py`, `tests/gdn/test_prefill_state_indices.py` | `implementation` + `author-benchmark` |
| **SGLang** (`sgl-project/sglang`) | `82ac83a8d0ab50ea7c9fe8e5b5934c2fa5a224bf` (`2026-10-20`, PR `#45339`) | `python/sglang/srt/models/deepseek_v4.py`, `deepseek_v4_platform_stage.py`, `deepseek_v4_backend_hip_radix.py` | `implementation` |
| **TensorRT-LLM** (`NVIDIA/TensorRT-LLM`) | `fe92fb923f12f98cfafc180bd9496126d2d9ef9e` (`2026-10-20`, PR `#19753`) & `4c0197d1e4410eb005748224ffd4c2ca5f0c7b44` (`2026-10-20`, PR `#20652`) | `cpp/tensorrt_llm/thop/moe/cutlass/moeOp.cpp`, `tensorrt_llm/_torch/moe/fused_moe/configurable_moe.py`, `tests/unittest/_torch/attention/test_cute_dsl_mla.py` | `implementation` |

---

## 2. FlashInfer SM90 GDN Prefill: Read-Only State Pool Scratch OOB Write (`#6849`)

### 2.1 Root Cause and Memory Address Arithmetic
In `chunk_gated_delta_rule`, when a caller supplies `state_indices`, `output_final_state=False`, and `output_state=None` (a read-only state pool `initial_state` of shape `[N_pool, H, V, K]`), the SM90 (`chunk_gated_delta_rule_sm90`) DSL kernel **unconditionally stores** each sequence's final state to `output_state[state_indices[i]]`.

Previously, the SM90 dispatcher allocated a compact `[num_seqs, H, V, K]` `float32` scratch tensor whenever `output_state is None`:
$$\text{Bytes}_{\text{slot}} = H \times V \times K \times 4 = 16 \times 128 \times 128 \times 4 = 1,048,576\text{ B} = 1\text{ MiB}$$
For `num_seqs = 2`, `N_pool = 5`, and `slots = [3, 0]`:
- The compact scratch buffer is $2 \times 1\text{ MiB} = 2\text{ MiB}$.
- Sequence 0 writes to `output_state[3]` ($[3\text{ MiB}, 4\text{ MiB})$), extending **$2\text{ MiB}$ past the end** of the $2\text{ MiB}$ buffer ($1\text{ MiB}$ of out-of-bounds global writes).
- Under `compute-sanitizer --tool memcheck` with `PYTORCH_NO_CUDA_MEMORY_CACHING=1` on H100, this produced **2,176 invalid global writes** before the fix and **0** after.

### 2.2 Two-Part Dispatch Fix
1. **No-Pool Index Drop**: If `initial_state is None and output_state is None`, `state_indices` is set to `None` so the kernel writes sequentially into the compact `[num_seqs, H, V, K]` scratch.
2. **Pool-Spanning Scratch**: When `state_indices is not None and initial_state is not None`, the temporary `output_state` spans `initial_state.shape` (`[N_pool, H, V, K]` = $5\text{ MiB}$), matching SM120.

---

## 3. SGLang AMD ROCm: DeepSeek-V4.1 Layer-20 Bounded-Replay Skip (`#45339`)

### 3.1 Fused mHC Boundary vs. Layer-20 KV Source Write
With `--enable-decoder-swa-bounded-replay`, DeepSeek-V4.1 decoder layers only need to execute heavy attention/MoE computation on the last $W_{\text{tail}} = 128$ prompt tokens per request during prefill. Layer 20 (`last_source = max(config.kv_source_layer_ids) = 20`) must still write KV cache entries for all $N_{\text{full}} = B \times L_{\text{prompt}}$ tokens.

On AMD ROCm (`_is_hip`), `hc_boundary_fused` delays the previous FFN's mHC post-processing to the start of the next layer. Consequently, ROCm previously set `tail_writes_source_kv = False` and `late_layer_start = 21`, running all of Layer 20 (indexer, attention, and MoE) across all $N_{\text{full}}$ tokens.

### 3.2 `fused_boundary_attn_input` Split and Token Savings
Commit `82ac83a8` introduces `fused_boundary_attn_input(layer, state)` in `deepseek_v4_platform_stage.py` to execute the pending `make_amd_hc_boundary` and `_input_norm`, calls `write_tail_source_kv` across all $N_{\text{full}}$ tokens, and advances `late_layer_start` to **Layer 20**:
- For $B=4, L_{\text{prompt}}=4096, W_{\text{tail}}=128$, Layer 20 heavy tokens drop from $16,384$ to $512$ (**$96.875\%$ reduction on Layer 20**);
- Across 40 decoder layers, total heavy token-layers drop from $36,352$ to $20,480$ (**$43.662\%$ total decoder reduction**).

---

## 4. TensorRT-LLM: CUTLASS MoE Compatibility Gates & MLA H=96 Batch Floor (`#19753`, `#20652`)

1. **OSS vs. Internal CUTLASS MoE Capability Checks (`#19753`)**:
   - When built without `USING_OSS_CUTLASS_MOE_GEMM`, `moe_host_src` (`TLLM_MOE_NO_GEMM`) enforces 5 early checks in `validateInternalMoeInputs`: `unpadded_hidden_size == hidden_size`, `num_valid_tokens == num_rows`, `!enable_alltoall`, `!has_input_sf || swizzled_input_sf`, and `!use_dynamic_fc2_scale`.
   - `ConfigurableMoE` passes `allow_alltoall=self.backend.supports_alltoall` to auto-select `AllGather/ReduceScatter` when All-to-All is unsupported, and wraps validation in `try ... finally` to call `self.comm.destroy()` on failure.
2. **SM103 MLA $H=96$ FP8 Single-Token Batch Floor (`#20652`)**:
   - `_PERF_MIN_BATCH_FP8[(96, 1)] = 8` routes single-token $H=96$ FP8 decode with `batch_size = 4` to `trtllm_gen` and `batch_size >= 8` (or multi-token verification `seq_len_q > 1`) to `cute_dsl_mla`.

---

## 5. Reproducible Offline Verification

```bash
python experiments/flashinfer_gdn_sm90_sglang_dsv41_trtllm_moe/run.py --check
```
