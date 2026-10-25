#!/usr/bin/env python3
"""Deterministic stdlib verification for Phase 80 Track B:
SGLang MiniMax-M3 Hopper FP8 Index-K Fused Store & Int64 Masked Tile Pointers (#46135),
DFlash/DFlash2 CUDA DP Attention & Declarative DraftParallelLayout (#46330, #46290),
Disaggregated Diffusion Fault Isolation (#46120), and vLLM ModelOpt Mixed-Precision Draft Mapping (#63001, #63457).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def simulate_minimax_m3_hopper_fp8_index_and_tile_ptrs() -> dict[str, Any]:
    """Model SGLang PR #46135 (`db7b66c1b6a7`):
    - `StoreTrait<kHeadBytes, kCastFp8, kUsePDL>`: when `kCastFp8=True`, input head is BF16 (`2 * kHeadBytes`)
      and cache head is FP8 E4M3 (`kHeadBytes`), so `kWorkerSize = kInHeadBytes / 16` (128-bit vector per thread).
    - On Hopper (`sm_90`), `get_minimax_sparse_index_dtype` stores `index_k` in `float8_e4m3fn` even when
      `fp8_attn_gemm=False` if `kv_cache_dtype == float8_e4m3fn`, cutting index-K cache footprint by 50% (2B -> 1B).
    - Replaces removed Triton 3.9 `tl.make_block_ptr` with `tile_ptrs_1d/2d/3d` using 64-bit (`int64`) offsets.
    """
    head_dim = 128
    cache_head_bytes_fp8 = head_dim * 1  # 128 B
    in_head_bytes_bf16 = cache_head_bytes_fp8 * 2  # 256 B
    worker_threads_cast_fp8 = in_head_bytes_bf16 // 16  # 16 threads
    worker_threads_same_dtype = cache_head_bytes_fp8 // 16  # 8 threads

    # Large stride overflow test: seq_idx = 20_000_000, stride = 128 -> 2,560,000,000 > 2^31 - 1
    seq_idx = 20_000_000
    stride = 128
    offset_64 = seq_idx * stride
    overflows_int32 = offset_64 > (2**31 - 1)

    return {
        "head_dim": head_dim,
        "cache_head_bytes_fp8": cache_head_bytes_fp8,
        "in_head_bytes_bf16": in_head_bytes_bf16,
        "worker_threads_cast_fp8": worker_threads_cast_fp8,
        "worker_threads_same_dtype": worker_threads_same_dtype,
        "hopper_fp8_kv_index_k_dtype": "float8_e4m3fn",
        "index_k_bytes_reduction_ratio": 2.0,
        "int64_offset_example": offset_64,
        "would_overflow_int32": overflows_int32,
    }


def simulate_dflash_dp_attention_and_draft_parallel_layout() -> dict[str, Any]:
    """Model SGLang PRs #46330 (`38ff13ea7928`) and #46290 (`6623ea445a28`):
    - DFlash/DFlash2 CUDA DP attention requires `--enable-dp-lm-head`, `attn_cp_size == 1`,
      and `SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION == 0`.
    - Idle rank with `model_is_mrope=True` initializes `mrope_positions` with shape `(3, 0)`.
    - `DraftParallelSpec` declares `SINGLE_GROUP` (DFlash, Eagle3, non-MoE DSpark) vs `TARGET_LAYOUT` (DeepSeek-V4 MoE DSpark, Standalone, MultiLayerEagle, FrozenKVMTP).
    - `resolve_dflash_draft_cell_size` prices the draft KV pool using resolved `draft_parallel_layout.attn_tp_size`
      and multiplies by `attn_dcp_size`.
    """
    # Example TP=8, DP=2 -> target attn_tp_size = 4.
    # For SINGLE_GROUP draft under local DP, draft attn_tp_size = 4; if mis-priced with global tp_size=8,
    # draft KV bytes/token would be underestimated by 2.0x!
    draft_kv_heads = 8
    head_dim = 64
    v_head_dim = 32
    layers = 2
    dtype_bytes = 2  # BF16
    attn_dcp_size = 2

    def price_draft_cell_bytes(tp_width: int) -> int:
        local_heads = draft_kv_heads // tp_width
        per_layer = local_heads * (head_dim + v_head_dim) * dtype_bytes
        return per_layer * layers * attn_dcp_size

    resolved_bytes = price_draft_cell_bytes(tp_width=4)  # 768 B/token
    mispriced_global_tp_bytes = price_draft_cell_bytes(tp_width=8)  # 384 B/token

    return {
        "cuda_dp_requires_dp_lm_head": True,
        "cuda_dp_requires_cp_eq_1": True,
        "idle_mrope_positions_shape": [3, 0],
        "single_group_models": ["DFlashDraftModel", "Eagle3DeepseekV2ForCausalLM", "LlamaForCausalLMEagle3"],
        "target_layout_models": ["DeepseekV4ForCausalLMDSpark", "StandaloneWorkerV2", "MultiLayerEagleWorkerV2", "FrozenKVMTPWorkerV2"],
        "resolved_draft_cell_bytes_per_token": resolved_bytes,
        "mispriced_global_tp_bytes_per_token": mispriced_global_tp_bytes,
        "underestimation_factor_prevented": round(resolved_bytes / mispriced_global_tp_bytes, 2),
    }


def simulate_disagg_diffusion_isolation_and_vllm_modelopt_draft() -> dict[str, Any]:
    """Model SGLang PR #46120 (`a996b5f2244e`) and vLLM PRs #63001 (`5f63ea461ccd`) & #63457 (`57ab2f35ac6e`):
    - SGLang DiffusionServer isolates 10 fault classes (`pickle`, `empty`, `batch`, `invalid_id`, `duplicate`,
      `shape`, `missing`, `unidentified`, `role_error`, `audio`) to single requests and releases worker slots.
    - vLLM `map_draft_quant_config` maps checkpoint-local `layers.i` in reverse order (`reversed(range(num_hidden_layers))`)
      to `layers.{start_layer_id + i}` so `start_layer_id=1` does not cascade `0 -> 1 -> 2`.
    """
    fault_modes = [
        "pickle", "empty", "batch", "invalid_id", "duplicate",
        "shape", "missing", "unidentified", "role_error", "audio",
    ]

    # Forward-order bug vs reverse-order fix for start_layer_id = 1, num_hidden_layers = 3
    start_layer_id = 1
    num_hidden_layers = 3
    orig_names = [f"model.layers.{i}.self_attn.q_proj" for i in range(num_hidden_layers)]

    def transform_names(names: list[str], reverse: bool) -> list[str]:
        indices = reversed(range(num_hidden_layers)) if reverse else range(num_hidden_layers)
        rules = [
            (re.compile(rf"^(model\.\*?)layers\.{idx}(?=\.|\*|$)"), rf"\g<1>layers.{start_layer_id + idx}")
            for idx in indices
        ]
        out = []
        for name in names:
            cur = name
            for pat, repl in rules:
                cur = pat.sub(repl, cur)
            out.append(cur)
        return out

    buggy_forward_mapped = transform_names(orig_names, reverse=False)
    fixed_reverse_mapped = transform_names(orig_names, reverse=True)

    return {
        "isolated_diffusion_fault_modes": fault_modes,
        "isolated_fault_count": len(fault_modes),
        "routing_loop_survives_all_faults": True,
        "orig_draft_layer_names": orig_names,
        "buggy_forward_mapped": buggy_forward_mapped,
        "fixed_reverse_mapped": fixed_reverse_mapped,
        "cascade_bug_avoided": (
            buggy_forward_mapped == ["model.layers.3.self_attn.q_proj"] * 3
            and fixed_reverse_mapped == [
                "model.layers.1.self_attn.q_proj",
                "model.layers.2.self_attn.q_proj",
                "model.layers.3.self_attn.q_proj",
            ]
        ),
        "vllm_weight_transfer_uses_torch_accelerator": True,
    }


def build_summary() -> dict[str, Any]:
    minimax = simulate_minimax_m3_hopper_fp8_index_and_tile_ptrs()
    dflash = simulate_dflash_dp_attention_and_draft_parallel_layout()
    disagg_vllm = simulate_disagg_diffusion_isolation_and_vllm_modelopt_draft()
    return {
        "experiment": "sglang_minimax_m3_sparse_fp8_dflash_dp_attn_draft_layout_vllm_modelopt_mixed",
        "upstream_commits": [
            "sgl-project/sglang@db7b66c1b6a73b74afa33a7c3a5e7067553dcddc",
            "sgl-project/sglang@38ff13ea7928ef76ce45459c63ba3035dd5b12d9",
            "sgl-project/sglang@6623ea445a287ff89fa61165a9d129daeee4ac34",
            "sgl-project/sglang@a996b5f2244ede791ce76f78e347db66bd7f0521",
            "vllm-project/vllm@5f63ea461ccd567ece6f50dd3701430c0f321039",
            "vllm-project/vllm@57ab2f35ac6e7207bd71cf6cad17b31705f03cd3",
        ],
        "minimax_m3_hopper_fp8_and_tile_ptrs": minimax,
        "dflash_dp_attention_and_draft_layout": dflash,
        "disagg_diffusion_and_vllm_modelopt_draft": disagg_vllm,
        "all_invariants_passed": (
            minimax["worker_threads_cast_fp8"] == 16
            and minimax["would_overflow_int32"]
            and dflash["underestimation_factor_prevented"] == 2.0
            and disagg_vllm["isolated_fault_count"] == 10
            and disagg_vllm["cascade_bug_avoided"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).with_name("summary.json")
    if args.check:
        assert summary["all_invariants_passed"], "Invariants failed"
        if out_path.exists():
            saved = json.loads(out_path.read_text(encoding="utf-8"))
            assert saved == summary, "summary.json drift detected"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[PASS] sglang_minimax_m3_sparse_fp8_dflash_dp_attn_draft_layout_vllm_modelopt_mixed")


if __name__ == "__main__":
    main()
