# SGLang BCG Shared/Caller Buffers & Ascend NPU Bitmask, vLLM Triton AV Draft Budget, Batch-Invariant WNA16, Jamba Prefix-Anchored Tool Streaming, Mooncake Descending Block Coalescing & TensorRT-LLM `@routes` Registry (Phase 98)

> **中文版本**: [sglang-bcg-nemotron-holder-lse-npu-bitmask-vllm-triton-av-wna16-jamba-mooncake-trtllm-phase98.md](sglang-bcg-nemotron-holder-lse-npu-bitmask-vllm-triton-av-wna16-jamba-mooncake-trtllm-phase98.md)
> **Priority**: `107` · **Question ID**: `sglang-bcg-nemotron-holder-lse-npu-bitmask-vllm-triton-av-wna16-jamba-mooncake-trtllm-phase98` · **Evidence Class**: `implementation` + `own-offline`

---

## 1. Executive Summary & Architectural Findings

Across 10 zero-overlap upstream commits in SGLang, vLLM, and TensorRT-LLM, Phase 98 establishes the following runtime and KV-transfer invariants:

1. **SGLang Breakable CUDA Graph (`bcg`) Caller-Allocated Buffer Contract (`sglang#46553`, `d4939cc267`; `#46551`, `5aeb6dfdb6`; `#46535`, `2c250563ae`)**:
   - `bcg.py` writes debug-mode forward output directly into `_shared_output_buffer` inside the eager break;
   - `nemotron_h.py` reads the prefill shared-expert output directly from the caller's holder without returning a separate reference;
   - `radix_attention.py` pre-allocates `lse = q.new_empty((q.shape[0], self.tp_q_head_num), dtype=torch.float32)` in the caller before `_eager_attention`, reducing extra allocations across 16 eager breaks from `48` to `0`.
2. **SGLang Ascend NPU Communicator Cleanup & Native `apply_token_bitmask` (`sglang#45950`, `3b058f28c2`; `#36074`, `fa622d311d`)**: Removes the redundant 71-line `NpuCommunicator` wrapper while documenting `aiter` IPC buffer staging for CUDA-graph capture, and wires `torch.ops.npu.apply_token_bitmask(logits, vocab_mask)` into `llguidance_backend.py` (`4,748` packed `int32` words for a `151,936` vocabulary).
3. **vLLM Single-Kernel Triton Adaptive Verification (AV) Draft-Budget Selection (`vllm#63421`, `41f5146b5b`)**: Replaces `torch.compile`'d `select_draft_token_budgets` with a fused Triton kernel (`_count_survival_at_least`), eliminating `~4.8 s` of Inductor compile warmup.
4. **vLLM Batch-Invariant WNA16 MoE Triton Guard (`vllm#61038`, `1fd6337e22`)**: Checks `if vllm_is_batch_invariant(): return False` at the top of `should_moe_wna16_use_cuda()` so `VLLM_BATCH_INVARIANT=1` never dispatches to the non-batch-invariant CUDA WNA16 kernel.
5. **vLLM Jamba Prefix-Anchored Tool Streaming & Mooncake Descending Block Coalescing (`vllm#55669`, `3f45273050`; `#63142`, `2841f08b4a`)**:
   - Anchors streamed Jamba tool arguments to `find_common_prefix` against already-emitted text so arguments started in the function-name delta preserve all subsequent deltas (`{"city": "Dallas", "state": "TX"}`);
   - Adds `_normalize_paired_block_order` in `mooncake_connector.py` to sort `(local_block_id, remote_block_id)` pairs ascending before kernel block expansion, coalescing 8 descending block pairs (`[9..2] -> [2..9]`) from `8` fragmented descriptors into `1` contiguous RDMA descriptor (`8.0x` reduction).
6. **TensorRT-LLM Declarative `@routes` Architecture Registry (`TensorRT-LLM#20789`, `158252825d`)**: Registers `modeling_v2` architecture routing resolvers via `@routes(...)` in `_router_index.py`.

---

## 2. Primary Source Evidence & Quantitative Metrics

| # | Evidence ID / Commit | Mechanism & Hardware Contract | Quantitative Gain / Correctness Invariant |
| :--- | :--- | :--- | :--- |
| 1 | `sglang-bcg-write-debug-mode-forward-output-buffer-inside-46553-d4939cc-2026` (`d4939cc267`) | `bcg.py`: copy debug forward output into `_shared_output_buffer` inside eager break | Eliminates temporary debug output buffers across BCG segments |
| 2 | `sglang-bcg-read-nemotron-h-prefill-shared-expert-output-46551-5aeb6df-2026` (`5aeb6dfdb6`) | `nemotron_h.py`: caller holder owns prefill shared-expert output | Zero returned tensor references across BCG graph boundaries |
| 3 | `sglang-bcg-write-radixattention-eager-break-lse-into-cal-46535-2c25056-2026` (`2c250563ae`) | `radix_attention.py`: caller pre-allocates `(tokens, tp_q_head_num)` FP32 LSE | Extra eager-break allocations reduced `48 -> 0` across 16 breaks |
| 4 | `sglang-cleanup-remove-redundant-npucommunicator-and-clar-45950-3b058f2-2026` (`3b058f28c2`) | `communicators.py` + `custom_all_reduce.py`: delete `NpuCommunicator` | `-71` redundant lines; clarifies IPC staging under graph capture |
| 5 | `sglang-npu-add-ascend-npu-bitmask-path-in-llguidance-bac-36074-fa622d3-2026` (`fa622d311d`) | `llguidance_backend.py`: `torch.ops.npu.apply_token_bitmask` | `4,748` `int32` words mask `151,936` vocab logits natively on NPU |
| 6 | `vllm-perf-spec-decode-replace-torch-compile-in-av-draf-63421-41f5146-2026` (`41f5146b5b`) | `rejection_sampler.py`: single-pass Triton `_count_survival_at_least` | Saves `~4.8 s` `torch.compile` warmup; deterministic budget count |
| 7 | `vllm-batch-invariance-keep-legacy-wna16-moe-on-triton-61038-1fd6337-2026` (`1fd6337e22`) | `fused_moe.py`: `if vllm_is_batch_invariant(): return False` | Preserves 100% deterministic Triton WNA16 MoE under batch invariance |
| 8 | `vllm-bugfix-anchor-jamba-streamed-tool-arguments-to-se-55669-3f45273-2026` (`3f45273050`) | `jamba_tool_parser.py`: `find_common_prefix` on streamed tool arguments | Lossless multi-chunk JSON argument reconstruction |
| 9 | `vllm-mooncake-coalesce-descending-block-ids-in-kvconne-63142-2841f08-2026` (`2841f08b4a`) | `mooncake_connector.py`: `_normalize_paired_block_order` before expansion | Coalesces 8 descending blocks from `8 -> 1` RDMA descriptor (`8.0x`) |
| 10 | `tensorrt-llm-feat-modeling-v2-register-architecture-routing-wi-20789-1582528-2026` (`158252825d`) | `_router_index.py`: `@routes(...)` decorator registry | Single source of truth for `modeling_v2` architecture routing |

---

## 3. Architect Design Exercises & Falsifiable Questions

1. **Calculation Exercise (LIFO Block Allocation & RDMA Descriptor Coalescing)**: When a LIFO free-block stack yields 8 consecutive blocks in descending order `[9, 8, 7, 6, 5, 4, 3, 2]` mapped to remote `[19, 18, 17, 16, 15, 14, 13, 12]`, `group_concurrent_contiguous` checks `src[i] == src[i-1] + 1` and emits 8 descriptors. Show why sorting the paired `(src, dst)` tuples by `src` ascending before sub-block expansion produces a single contiguous transfer without altering KV placement semantics.
2. **Falsifiable Hypothesis**: Does switching small batches (`B <= 16`) from Triton WNA16 to CUDA WNA16 preserve bitwise batch invariance when `VLLM_BATCH_INVARIANT=1`? Source and numerical verification prove **No**: different reduction tree orders across the two kernels break cross-batch determinism.

---

## 4. Reproducible Offline Experiment

- **Experiment Script**: `experiments/sglang_bcg_nemotron_holder_lse_npu_bitmask_vllm_triton_av_wna16_jamba_mooncake_trtllm_phase98/run.py`
- **Verification Command**: `python experiments\sglang_bcg_nemotron_holder_lse_npu_bitmask_vllm_triton_av_wna16_jamba_mooncake_trtllm_phase98\run.py --check`
- **Unit Test**: `tests/test_phase98_experiments.py`
