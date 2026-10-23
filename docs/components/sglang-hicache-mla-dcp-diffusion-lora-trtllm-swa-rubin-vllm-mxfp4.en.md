# SGLang HiCache MLA DCP L3 Write Deduplication & Sparse Trailing-Page Bitmap Voting, Diffusion Dynamic LoRA Guard, TensorRT-LLM SWA Branch Snapshots & Rubin SM107, and vLLM Humming MXFP4 Override & Rust Wire Parity

> **Phase 56 (`priority: 65`) · Primary Source & Commit-Level Deep Dive**
> - `sgl-project/sglang@c677171c1df08ef20da92c8ba6be12507c92a248` (`#41187`)
> - `sgl-project/sglang@02b983603dc743d2d37eb34410d7e3cb0b238a8c` (`#45943`)
> - `sgl-project/sglang@0e3496288488ff4c70aa22d0b44643f5d1ca2b86` (`#45966`)
> - `NVIDIA/TensorRT-LLM@ae1aa43964947a272ef170933b3b9f5194a5955d` (`#19877`)
> - `NVIDIA/TensorRT-LLM@b0af321b0c9372fa6e80501c51dc9d5b6da380f2` (`#20334`)
> - `NVIDIA/TensorRT-LLM@854ccee8a944a7597149a96421d92f9dd5496579` (`#20167`)
> - `NVIDIA/TensorRT-LLM@448321fb98261098a3eb5bd319f896959dc7b676` (`#20605`)
> - `NVIDIA/TensorRT-LLM@d7ec5162ac8d935b21bb5eac719e5eac99664cec` (`#20242`)
> - `vllm-project/vllm@97b6a67d94f7078221dcc9839141a0e2b5ff9a20` (`#62751`)
> - `vllm-project/vllm@7b1d07440dde8426a8b3d85e5f5815c06913eb82` (`#63295`)
> - `vllm-project/vllm@4a5ee8fc77519a5fc1c68700809ac972277d8f6a` (`#63316`)
> - `vllm-project/vllm@933f7962aeb81aa581eadb24dedd746fab20b2dd` (`#63317`)

---

## 1. SGLang HiCache MLA + Decode Context Parallelism (`dcp_size > 1`) L3 Storage & Diffusion Dynamic LoRA (`#41187`, `#45943`, `#45966`)

- **HiCache MLA DCP L3 Page Mapping & Writer Deduplication (`#41187`)**:
  - Under Decode Context Parallelism (`dcp_size > 1`), logical page `k` spans `logical_page_size = page_size * dcp_size` tokens while occupying 1 physical storage page per DCP rank (`_storage_page_index(index) = index // dcp_size`).
  - **Cross-TP Writer Deduplication**: Because MLA KV cache is replicated across TP ranks that share the same `dcp_rank`, `HiCacheController` sets `is_storage_writer = (not is_mla_model) or (tp_rank == dcp_rank)`. For `TP=4, DCP=2`, only ranks `0` and `1` write `dcp0` and `dcp1` to L3 (`file` / `mooncake`) while ranks `2` and `3` set `backup_skip = True` and read those shared objects, cutting L3 write amplification by **`tp_size / dcp_size = 2x`**.
  - **Sparse Trailing-Page Checkpoint Bitmap Voting (`PoolHitPolicy.TRAILING_PAGES`)**: For hybrid models with Mamba/KDA state checkpoints at sparse page boundaries (e.g., pages 512 and 1024, but not 768), reducing a scalar `MIN(storage_hit_count)` across ranks can land on a non-checkpoint page. `_sync_prefetch_hit_query` instead reduces a bitmask vector `hits[0..num_pages]` via elementwise `ReduceOp.MIN` across ranks.
- **Diffusion `FUSED_LINEAR_GELU` Dynamic LoRA Guard & LTX-2 Host-Sync Elimination (`#45943`, `#45966`)**:
  - `can_use_linear_gelu` unwraps `BaseLayerWithLoRA` during static capability checks so `FUSED_LINEAR_GELU` remains mounted, but returns `False` per call whenever `linear.lora_active` is true so active dynamic LoRA adapters are never silently dropped by the cublasLt base-weight epilogue.
  - LTX-2 denoising pre-materializes `sigmas_host` (`cpu`) and `sigmas_device` (`cuda`) once per loop and moves `encoder_attention_mask` construction outside the transformer block loop, eliminating per-step `.item()` / `.to(device)` stream syncs.

---

## 2. TensorRT-LLM SWA Branch Snapshots, Low-M GEMM Capture Safety, Rubin SM107 & Context MLA Separate-V Stride (`#19877`, `#20334`, `#20167`, `#20605`, `#20242`)

- **SWA Branch Snapshots (`#19877`)**: `_swa_branch_snapshot_point` computes `point = (divergence // tokens_per_block) * tokens_per_block` when `divergence < num_lookup_tokens` and `reused < point < prompt_len`, switching the executor to `(ContextChunkingPolicy.FORCE_CHUNK, chunk_unit_size)` so prefill stops at the divergence boundary and materializes a reusable Radix block snapshot before SWA evicts older blocks.
- **Low-M GEMM CUDA Graph Capture Safety (`#20334`)**: Allocates a fresh `torch.empty((m, n))` buffer whenever `torch.cuda.is_current_stream_capturing()` is true instead of reusing `self._output_buffers[(module_key, m_bucket, n)]`, preventing multiple in-graph calls from overwriting earlier outputs.
- **Rubin (`SM107`) MiniMax-M3 Enablement (`#20167`) & Multimodal `DecoderLoraOwner` (`#20605`)**: Adds `SM107` to `EXTRA_PAGE_SIZES_ONLY_SMS = frozenset({107})` so 128-token pages (Eagle3 draft) route to TRTLLM-Gen FMHA while 32-token pages use fallback FMHA; zeroes KV/scale/index buffers at init to avoid `0 * NaN = NaN` in masked tail slots; scopes multimodal LoRA loading to `DecoderLoraOwner` (`language_model` or `model`).
- **Context MLA Separate-V Token Stride (`#20242`)**: Plumbs `v_stride_in_bytes` / `v_token_stride_elements` (enforcing 16-byte TMA alignment) through `AttentionOp`, `FusedMHARunnerV2`, `quantizeCopyInputToFp8Kernel`, and `TllmGenFmhaKernel`.

---

## 3. vLLM Humming MXFP4 Activation Override & Rust Frontend Wire Parity (`#62751`, `#63295`, `#63316`, `#63317`)

- **Humming MXFP4 MoE Activation Override (`#62751`)**: `_humming_input_schema` resolves `_resolve_activation_key(model_activation_key)` for both `gpt_oss_mxfp4` and `mxfp4` checkpoints so `--quantization-config.moe.activation mxfp8` takes effect.
- **Rust `LoRARequest` 6-Field Tuple & `MooncakeTransferStats` (`#63295`, `#63316`, `#63317`)**: Drops `tensorizer_config_dict` from Rust `LoraRequest` to match Python's 6-element `msgspec.Struct(array_like=True)` wire format; introduces `LoadedLora` so cached inference requests always set `load_inplace = false`; places `MooncakeTransfer` before `Multi` in `#[serde(untagged)] enum KvConnectorStats`.

---

## 4. Deterministic Verification

Run `python experiments/sglang-hicache-mla-dcp-diffusion-lora-trtllm-swa-rubin-vllm-mxfp4/run.py --check` to verify all invariants (`summary.json`).
