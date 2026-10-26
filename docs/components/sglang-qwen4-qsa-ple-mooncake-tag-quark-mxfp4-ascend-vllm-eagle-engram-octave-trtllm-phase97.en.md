# SGLang Qwen4 QSA/PLE Caller-Held Bridges, Mooncake Session Tagging, Quark MXFP4 Shared Expert Fusion, Ascend D2H-Free DSA, vLLM Finalized Prefix Snapshot, EAGLE Lookahead Block Hashes, DSv4.1 Peer-HBM Engram, ROCm Octave 3-Bit KV & TRT-LLM Deterministic mHC (Phase 97)

> **中文版**: [sglang-qwen4-qsa-ple-mooncake-tag-quark-mxfp4-ascend-vllm-eagle-engram-octave-trtllm-phase97.md](sglang-qwen4-qsa-ple-mooncake-tag-quark-mxfp4-ascend-vllm-eagle-engram-octave-trtllm-phase97.md)
> **Priority**: `106` · **Question ID**: `sglang-qwen4-qsa-ple-mooncake-tag-quark-mxfp4-ascend-vllm-eagle-engram-octave-trtllm-phase97` · **Evidence Class**: `implementation` + `own-offline`

---

## 1. Executive Summary & Architectural Findings

Phase 97 analyzes 10 zero-overlap commits across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:

1. **Caller-Held Pre-Allocated QSA Top-K & PLE Bridge Buffers (`sglang#46543`, `5e420a1258`)**: `qwen4_exp` writes QSA top-k indices and Per-Layer Embedding (PLE) batch breaks directly into caller-held pre-allocated output tensors (`_qsa_prefill_topk_bridges`) so breakable CUDA Graph segments never return newly allocated temporary tensors across segment boundaries.
2. **Short-Circuiting `is_cpu()` Before `platform.machine()` (`sglang#46640`, `5a1f174700`)**: Checking `os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1"` before `is_host_cpu_x86() / is_host_cpu_arm64()` eliminates `os.uname()` graph breaks under `torch.compile(fullgraph=True)` in `RMSNorm.forward_native`.
3. **Session-Tagged Mooncake & UMBP KV Offloads (`sglang#44611`, `5987ec82ed`)**: `MooncakeDirectLinker` and `UMBPDirectLinker` propagate `HiCacheStorageExtraInfo.session_id` into `ReplicateConfig` when supported by the transfer engine.
4. **On-Load Quark BF16 Shared-Expert Quantization to MXFP4 (`sglang#41899`, `0cd2a63b36`)**: Under `--enforce-shared-experts-fusion`, SGLang quantizes BF16 shared experts in AMD Quark MXFP4 Qwen3.5/Qwen3.8 checkpoints to OCP MXFP4 (`3.7647x` compression including E8M0 block scales), fusing shared and routed experts into a single MXFP4 MoE kernel launch.
5. **D2H-Free Block-Table Gather in Ascend NPU Mirrorless DSA Graph Replay (`sglang#45949`, `d6ccb495da`)**: Uses `ForwardBatch.max_seq_len_host` instead of device `.max().item()` to slice `block_table`, reducing per-step D2H synchronizations from `1` to `0`.
6. **Freezing Processed Prefix Token Counts at Request Finalization (`vllm#57646`, `c13d4a5a46`)**: `_VllmReqContext` snapshots `_RequestTokens` (`num_computed_tokens`, `num_in_flight_tokens`, `num_prompt_tokens`, `num_tokens`) upon finalization so asynchronous KV offload handlers observe exact processed prefix lengths after preemption or abort.
7. **Lookahead Block Hashes for EAGLE Speculative Prefix Caching (`vllm#61589`, `2fb320d830`)**: `BlockStored` KV events attach `lookahead_token_ids: list[list[int]] | None` so EAGLE draft tokens complete boundary blocks for prefix hash calculation while preserving MessagePack wire compatibility.
8. **Direct Peer-HBM Engram Reads for DeepSeek-V4.1 (`vllm#62091`, `85e199e60c`)**: Reads unowned GPU Engram rows (`peer_rows`) directly from peer HBM over IPC/NVLink instead of running DP collectives (`8.0x` communication reduction at `DP=8`).
9. **Native 3-Bit `ROCM_OCTAVE` KV Cache Backend (`vllm#59774`, `00b6c4c565`)**: Adds `ROCM_OCTAVE` (`VLLM_OCTAVE_KV` on `gfx942`/`gfx1100`), achieving `5.3333x` bitwidth compression over FP16/BF16.
10. **Deterministic Mode Enforcement in TensorRT-LLM `mhc_fused_hc` (`TensorRT-LLM#20228`, `8ea175e248`)**: Bypasses non-deterministic autotuned tactics when `torch.are_deterministic_algorithms_enabled()` or `FORCE_DETERMINISTIC` is active.

---

## 2. Primary Source Evidence Matrix

| # | Source ID / Commit | Mechanism & Contract | Quantitative Result / Invariant |
| :--- | :--- | :--- | :--- |
| 11 | `sglang-qwen4-qsa-topk-ple-caller-held-outputs-pr46543-2026` (`5e420a1258`) | Caller-held `_qsa_prefill_topk_bridges` in `qwen4_exp.py` | Breakable CUDA Graph compatible QSA/PLE bridge writes |
| 12 | `sglang-fix-cpu-engine-detection-torch-compile-fullgraph-pr46640-2026` (`5a1f174700`) | Env-first short-circuit in `is_cpu()` | `0` `os.uname()` calls on GPU path; `fullgraph=True` clean |
| 13 | `sglang-mem-cache-tag-mooncake-offloads-session-pr44611-2026` (`5987ec82ed`) | `session_id` propagation to Mooncake `ReplicateConfig` | Session-aware replica placement for KV offloads |
| 14 | `sglang-amd-quark-qwen38-quantize-bf16-shared-expert-mxfp4-pr46342-2026` (`0cd2a63b36`) | On-load `quark_moe_bf16_to_mxfp4` for BF16 shared experts | `3.7647x` shared-expert weight compression + single fused MoE launch |
| 15 | `sglang-npu-ascend-d2h-free-blocktable-gather-dsa-graph-pr45949-2026` (`d6ccb495da`) | `ForwardBatch.max_seq_len_host` block-table sizing | `1 -> 0` D2H syncs/step in mirrorless DSA graph replay |
| 16 | `vllm-kv-offload-expose-processed-prefix-finalization-pr57646-2026` (`c13d4a5a46`) | Frozen `_RequestTokens` snapshot on `_VllmReqContext` | Preserves exact processed prefix (`3072` tokens) after scheduler cleanup |
| 17 | `vllm-spec-decode-lookahead-block-hashes-eagle-prefix-pr61589-2026` (`2fb320d830`) | `BlockStored.lookahead_token_ids` for EAGLE draft tokens | Completes 16-token block hashes across draft boundaries |
| 18 | `vllm-dsv41-read-engram-tables-peer-hbm-instead-dp-pr62091-2026` (`85e199e60c`) | Peer-HBM IPC `peer_rows` gather in `deepseek_v4_engram.py` | `8.0x` communication reduction at `DP=8` vs collective gather |
| 19 | `vllm-rocm-octave-kv-native-3bit-kv-cache-backend-pr59774-2026` (`00b6c4c565`) | `ROCM_OCTAVE` 3-bit KV cache (`gfx942`/`gfx1100`) | `5.3333x` bitwidth compression vs FP16 |
| 20 | `trtllm-honor-deterministic-mode-fused-mhc-dispatch-pr20228-2026` (`8ea175e248`) | Deterministic guard in `mhc_fused_hc` | Bypasses non-deterministic autotuned tactics |

---

## 3. Reproducible Offline Verification

- **Script**: `experiments/sglang_qwen4_qsa_ple_mooncake_tag_quark_mxfp4_ascend_vllm_eagle_engram_octave_trtllm_phase97/run.py`
- **Command**: `python experiments\sglang_qwen4_qsa_ple_mooncake_tag_quark_mxfp4_ascend_vllm_eagle_engram_octave_trtllm_phase97\run.py --check`
- **Unit Test**: `tests/test_phase97_experiments.py`
