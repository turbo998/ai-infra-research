"""Phase 120 Track B stdlib experiment: SGLang SM107 2-Shot AllReduce,
DeepSeek-V4.1 Engram DP Lookup Auto-Collective, Triton 3.9 Swap-AB Scale Broadcast,
Split-Expert MXFP4 Marlin MoE, DSV4 Draft Metadata Glue Guard, vLLM Kimi-K3 ROCm Folded Weight,
Spec-Decode LoRA Logits Alignment, HiSparse State-Slot Residency, KVCR Prefix Recency & TRT-LLM MoE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_sglang_sm107_2shot_allreduce() -> dict[str, Any]:
    """Model SGLang 08b9ec6: SM107 (Rubin VR) custom all-reduce 2shot_push thresholds."""
    kb = 1024
    mb = 1024 * 1024
    configs = {
        "tp4": {
            "old_graph_1shot_max_bytes": int(6.938 * mb),
            "new_graph_1shot_max_bytes": int(1.000 * mb),
            "old_eager_1shot_max_bytes": int(9.812 * mb),
            "new_eager_1shot_max_bytes": int(1.000 * mb),
            "two_shot_push_threshold_bytes": 16 * mb,
            "mc_blocks": 128,
        },
        "tp8": {
            "old_1shot_max_bytes": int(2.438 * mb),
            "new_1shot_max_bytes": 384 * kb,
            "two_shot_push_threshold_bytes": 16 * mb,
            "mc_blocks": 96,
        },
        "tp16": {
            "old_1shot_max_bytes": 832 * kb,
            "new_1shot_max_bytes": 256 * kb,
            "two_shot_push_threshold_bytes": 16 * mb,
            "mc_blocks": 32,
        },
    }
    return {
        "arch": "sm_107",
        "configs": configs,
        "tp8_1shot_threshold_reduction_ratio": round((2.438 * mb) / (384 * kb), 4),
        "tp16_1shot_threshold_reduction_ratio": round((832 * kb) / (256 * kb), 4),
    }


def cols_per_rank(columns: int, tp_size: int) -> int | None:
    return columns // tp_size if columns % tp_size == 0 else None


def model_sglang_dsv41_engram_dp_collective() -> dict[str, Any]:
    """Model SGLang e35b5bc: DeepSeek-V4.1 Engram DP lookup auto collective selection."""
    # DeepSeek-V4.1 has 24 hash columns (e.g. engram_n_heads * (engram_max_ngram_size - 1) = 24)
    total_columns = 24
    tp_Evaluations = []
    for tp_size in [2, 3, 4, 5, 6, 8, 12, 16]:
        k = cols_per_rank(total_columns, tp_size)
        selected_eager_prefill = "all_to_all" if k is not None else "all_reduce"
        # Relative bytes moved per rank for N total tokens and hidden dim D:
        # reduce_scatter moves (tp_size - 1)/tp_size * N * columns * D;
        # all_to_all moves (tp_size - 1)/tp_size * N * (columns / tp_size) * D -> 1/tp_size of reduce_scatter
        byte_ratio_vs_reduce_scatter = round(1.0 / tp_size, 4) if k is not None else 1.0
        tp_Evaluations.append(
            {
                "tp_size": tp_size,
                "cols_per_rank": k,
                "selected_eager_prefill": selected_eager_prefill,
                "byte_ratio_vs_reduce_scatter": byte_ratio_vs_reduce_scatter,
            }
        )
    return {
        "total_hash_columns": total_columns,
        "divisible_tp_sizes": [e["tp_size"] for e in tp_Evaluations if e["cols_per_rank"] is not None],
        "evaluations": tp_Evaluations,
        "author_measured_wins_out_of_144": 121,
        "author_speedup_range_8k_plus_tokens": [1.06, 1.60],
    }


def model_sglang_moe_and_dsv4_fixes() -> dict[str, Any]:
    """Model SGLang faa56e2 (Triton 3.9 swap_ab), bb4bb01 (split-expert MXFP4 Marlin),
    and 2eab19a (dsv4 draft metadata glue host lengths guard).
    """
    return {
        "triton39_swap_ab_scalar_broadcast_expr": "accumulator += tl.dot(a, b) * (a_scale * b_scale[None, :])",
        "mxfp4_marlin_group_size": 32,
        "mxfp4_marlin_group_blocks": 2,
        "mxfp4_marlin_scale_type": "kFE8M0fnu",
        "nvfp4_marlin_scale_type": "kFE4M3fn",
        "mxfp4_dequant_skip_flop": False,
        "nvfp4_dequant_skip_flop": True,
        "dsv4_glue_refuses_when_needs_cpu_seq_lens": True,
    }


def model_vllm_spec_lora_hisparse_kvcr_and_trtllm() -> dict[str, Any]:
    """Model vLLM b71c1d5 (Kimi-K3 ROCm), 06e172b (MRV2 spec-decode LoRA),
    66f1d0f (HiSparse state-slot residency), e3883b7 (KVCR prefix recency),
    and TRT-LLM f4bc8ee (B300 block-scale MoE re-enable).
    """
    # Spec decode MRV2 LoRA logits alignment:
    # Suppose 3 requests with lora_ids [10, 20, 30] and num_logits [1, 4, 3] (drafts 0, 3, 2)
    lora_ids = [10, 20, 30]
    cu_num_logits = [0, 1, 5, 8]
    num_logits = [cu_num_logits[i + 1] - cu_num_logits[i] for i in range(len(lora_ids))]
    old_prompt_lora_mapping = list(lora_ids)  # length 3 != 8 logits rows!
    new_prompt_lora_mapping = [
        lid for lid, count in zip(lora_ids, num_logits) for _ in range(count)
    ]

    # HiSparse residency layout transpose: [resident_groups, max_num_reqs, max_num_pages]
    # Request batch row 0 maps to state slot 5
    request_state_indices = [5, 2]
    batch_req_idx = 0
    remapped_state_slot = max(request_state_indices[batch_req_idx], 0)

    # KVCR on_request_finished processed_tokens formula:
    # processed_tokens = min(max(0, computed_tokens - in_flight), max(prompt_tokens, num_tokens - 1))
    computed_tokens = 128
    in_flight = 16
    prompt_tokens = 64
    num_tokens = 115
    processed_tokens = min(
        max(0, computed_tokens - in_flight),
        max(prompt_tokens, num_tokens - 1),
    )
    key_positions = {"b0": 16, "b1": 64, "b2": 112, "b3": 128}
    touched_keys = [k for k, pos in key_positions.items() if pos <= processed_tokens]

    return {
        "kimi_k3_rocm_forwards_process_weights_after_loading": True,
        "spec_lora_old_len": len(old_prompt_lora_mapping),
        "spec_lora_new_len": len(new_prompt_lora_mapping),
        "spec_lora_total_logits_rows": cu_num_logits[-1],
        "spec_lora_mapping_matches_logits_rows": len(new_prompt_lora_mapping) == cu_num_logits[-1],
        "hisparse_residency_shape_order": ["resident_groups", "max_num_reqs", "max_num_pages"],
        "hisparse_remapped_state_slot": remapped_state_slot,
        "kvcr_processed_tokens": processed_tokens,
        "kvcr_touched_keys": touched_keys,
        "trtllm_b300_reenabled_cases_lines_removed": 113,
        "trtllm_routes_in_kernel_guard": "trtllm_use_router_logits and not backend._routes_outside_the_kernel()",
    }


def run_all() -> dict[str, Any]:
    return {
        "phase": 120,
        "track": "track_b",
        "title": "SGLang SM107 2-Shot AllReduce, Engram DP Auto-Collective, Split-Expert MXFP4 Marlin, vLLM Spec-LoRA, HiSparse State-Slot & KVCR Prefix Recency",
        "sm107_2shot_allreduce": model_sglang_sm107_2shot_allreduce(),
        "dsv41_engram_dp_collective": model_sglang_dsv41_engram_dp_collective(),
        "sglang_moe_and_dsv4_fixes": model_sglang_moe_and_dsv4_fixes(),
        "vllm_and_trtllm_fixes": model_vllm_spec_lora_hisparse_kvcr_and_trtllm(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_all()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.check:
        assert summary["sm107_2shot_allreduce"]["tp8_1shot_threshold_reduction_ratio"] == 6.5013
        assert summary["dsv41_engram_dp_collective"]["divisible_tp_sizes"] == [2, 3, 4, 6, 8, 12]
        assert summary["vllm_and_trtllm_fixes"]["spec_lora_mapping_matches_logits_rows"] is True
        assert summary["vllm_and_trtllm_fixes"]["kvcr_processed_tokens"] == 112
        assert summary["vllm_and_trtllm_fixes"]["kvcr_touched_keys"] == ["b0", "b1", "b2"]

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
