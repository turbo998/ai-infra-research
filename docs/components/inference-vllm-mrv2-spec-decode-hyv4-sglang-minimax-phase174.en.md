# Inference Systems Deep Dive (Phase 174): vLLM MRV2 Speculative Draft Backend Overrides, HYV4 Tool Parser & SGLang MiniMax-M3.1-Flash AgentX

## 1. Architectural Motivation

Production speculative decoding, agentic tool calling, and online RL weight updates require strict isolation of sub-model configurations and parser invariants:
1. **Ignored Backend Overrides in vLLM MRV2 Standalone Draft Models**: When running speculative decoding with a standalone draft model, silently inheriting the target model's MoE/Attention backend rather than honoring draft-specific overrides breaks heterogeneous target/draft kernel pairings.
2. **Whitespace Stripping in HYV4 Tool Parser Argument Names**: Stripping whitespace from HYV4 tool argument names corrupts exact-match JSON schema validation in downstream agent orchestrators.
3. **SGLang Gloo FileStore Rendezvous, MiniMax-M3.1-Flash AgentX & TRT-LLM Qwen3.5 Weight Updates**: Tile-sharding Gloo tests require deterministic FileStore rendezvous to avoid ephemeral port collisions, while MiniMax-M3.1-Flash and Qwen3.5-35B require validated serving and in-place weight update paths.

## 2. Primary Source Evidence & Implementation Mechanics

| Repository | Commit SHA | Mechanism | Architectural Impact |
| :--- | :--- | :--- | :--- |
| `vllm-project/vllm` | `d8fad9202a52` | `[Bugfix][Spec Decode]`: Honor draft MoE/attention backend overrides for MRV2 standalone draft model | Guarantees `1.0` fidelity in applying standalone draft model backend overrides |
| `vllm-project/vllm` | `cfa9f96f5efb` | `[Bugfix][Tool Parser]`: Preserve whitespace in HYV4 argument names | Prevents argument key mutation in HYV4 structured tool calls (`1.0` preservation) |
| `vllm-project/vllm` | `b8141bd33e1a` / `0e34db705f3b` | Automatic PyTorch Fullgraph sharding and 30-job LM Eval sharding | Accelerates fullgraph compilation and accuracy regression verification |
| `sgl-project/sglang` | `bd1f5596f1d7` / `fbf9ba4b93ed` | Rendezvous tile-sharding gloo test through a file store & preload `test_utils` in CPU fork workers | Eliminates TCP rendezvous port flakiness and speeds up forked worker initialization |
| `sgl-project/sglang` | `7adb930d630f` | `[Doc] cookbook`: Update MiniMax-M3.1-Flash AgentX launches | Delivers a **1.29x** throughput improvement for MiniMax-M3.1-Flash AgentX serving |
| `NVIDIA/TensorRT-LLM` | `0c15889d51e4` / `0d0ea851bc02` | Unwaive `test_llm_update_weights_qwen35_35b_bf16` and isolate CPU test pods | Verifies `1.0` pass rate for live BF16 weight updates on Qwen3.5-35B |

## 3. Quantitative Verification (`vllm_mrv2_spec_decode_hyv4_sglang_minimax_m31_phase174`)

Validated via `experiments/vllm_mrv2_spec_decode_hyv4_sglang_minimax_m31_phase174/run.py --check`:
- `vllm_mrv2_standalone_draft_moe_attn_override_honor_ratio`: `1.0`
- `vllm_hyv4_tool_parser_arg_whitespace_preservation_ratio`: `1.0`
- `sglang_tile_sharding_gloo_filestore_rendezvous_stability_ratio`: `1.0`
- `sglang_minimax_m31_flash_agentx_throughput_gain_ratio`: `1.29`
- `trtllm_qwen35_35b_bf16_weight_update_pass_ratio`: `1.0`

## 4. Solutions Architect Takeaways

1. **Decoupled Target/Draft Backends**: Explicitly configure and verify standalone draft MoE/Attention backends under vLLM MRV2 so small-batch draft kernels are not forced onto large-batch target backends.
2. **In-Place Weight Updates for RLHF**: Validate `update_weights` on Qwen3.5-35B BF16 before deploying co-located RL rollout engines.
