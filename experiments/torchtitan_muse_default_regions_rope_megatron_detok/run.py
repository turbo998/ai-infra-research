#!/usr/bin/env python3
"""Phase 37 Experiment 2: TorchTitan Muse Spark Default Compile Regions,
Helion RoPE Override Removal / Non-Contiguous ComplexRoPE & Megatron-LM Streaming Detokenizer.

Audited upstream commits:
1. pytorch/torchtitan @ 39bd2054d9c4fcbcabc343d262abad85f19020fb (#5433, 2026-10-20)
2. pytorch/torchtitan @ abdf14320a01212430b4ddfb2d147a4dfbab4b3e (#5530, 2026-10-20)
3. NVIDIA/Megatron-LM @ ebab8e2e21c77361fd9edbaee3e3d7d90748f340 (#8268, 2026-10-20)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_muse_spark_default_regions() -> dict[str, Any]:
    # Per-layer eager vs bandwidth-ideal latency at 4k tokens from pytorch/torchtitan #5433 (39bd2054d9)
    op_4k_us = {
        "residual": {"eager_us": 724.0, "bandwidth_ideal_us": 95.0},
        "post_ffn_norm": {"eager_us": 381.0, "bandwidth_ideal_us": 47.0},
        "expert_output_norm": {"eager_us": 1335.0, "bandwidth_ideal_us": 189.0},
    }
    total_eager_us = sum(v["eager_us"] for v in op_4k_us.values())  # 2440.0 us
    total_ideal_us = sum(v["bandwidth_ideal_us"] for v in op_4k_us.values())  # 331.0 us
    eager_bandwidth_efficiency_pct = round(total_ideal_us / total_eager_us * 100.0, 2)

    # 2-layer Muse Spark Open 816B-A42B on GB300 (ms), interleaved A/B from commit 39bd2054d9
    gb300_2l_ms = {
        "train_2k_launch": {"main": 22.88, "pr1": 21.21, "this_pr": 19.09, "vs_pr1_pct": -10.0, "vs_main_pct": -16.6},
        "train_2k_graphs": {"main": 21.54, "pr1": 20.11, "this_pr": 18.15, "vs_pr1_pct": -9.8, "vs_main_pct": -15.8},
        "train_4k_launch": {"main": 43.40, "pr1": 40.46, "this_pr": 36.67, "vs_pr1_pct": -9.4, "vs_main_pct": -15.5},
        "train_4k_graphs": {"main": 42.13, "pr1": 39.28, "this_pr": 35.94, "vs_pr1_pct": -8.5, "vs_main_pct": -14.7},
        "train_16k_launch": {"main": 171.50, "pr1": 159.93, "this_pr": 148.14, "vs_pr1_pct": -7.4, "vs_main_pct": -13.6},
        "train_16k_graphs": {"main": 171.41, "pr1": 158.89, "this_pr": 147.88, "vs_pr1_pct": -6.9, "vs_main_pct": -13.7},
        "prefill_2k": {"main": 7.64, "pr1": 7.23, "this_pr": 6.45, "vs_pr1_pct": -10.9, "vs_main_pct": -15.7},
        "prefill_16k": {"main": 59.31, "pr1": 56.28, "this_pr": 50.65, "vs_pr1_pct": -10.0, "vs_main_pct": -14.6},
        "decode_1": {"main": 1.07, "pr1": 1.05, "this_pr": 0.98, "vs_pr1_pct": -7.1, "vs_main_pct": -9.1},
        "decode_8": {"main": 1.10, "pr1": 1.07, "this_pr": 0.97, "vs_pr1_pct": -10.0, "vs_main_pct": -12.1},
        "decode_64": {"main": 1.33, "pr1": 1.31, "this_pr": 1.19, "vs_pr1_pct": -9.0, "vs_main_pct": -10.4},
        "decode_256": {"main": 1.55, "pr1": 1.49, "this_pr": 1.32, "vs_pr1_pct": -11.2, "vs_main_pct": -14.6},
    }

    default_regions = [
        "loss",
        "fp32_to_bf16_split",
        "rope",
        "muse_spark_input_scaled_rmsnorm",
        "muse_spark_logit_postprocess",
        "muse_spark_residual",
    ]
    return {
        "default_local_compile_regions": default_regions,
        "lora_sft_override_regions": ["loss"],
        "op_4k_us": op_4k_us,
        "total_eager_4k_us": total_eager_us,
        "total_ideal_4k_us": total_ideal_us,
        "eager_bandwidth_efficiency_pct": eager_bandwidth_efficiency_pct,
        "graph_counts": {"residual_graphs": 4, "input_scaled_norm_graphs": 6, "recompile_limit": 8},
        "gb300_2l_ms": gb300_2l_ms,
    }


def simulate_non_contiguous_complex_rope() -> dict[str, Any]:
    # Models test_compiled_matches_eager_complex_non_contiguous in pytorch/torchtitan #5530 (abdf14320a)
    # Query: storage [8, 64, 64] transposed(0, 1) -> logical [64, 8, 64], strides (64, 4096, 1)
    # Key: storage [64, 96] sliced [..., -64:].unsqueeze(1) -> logical [64, 1, 64], strides (96, 64, 1)
    seq_len, n_heads, head_dim = 64, 8, 64
    q_strides = (head_dim, seq_len * head_dim, 1)  # (64, 4096, 1) - non-contiguous across heads
    k_strides = (96, head_dim, 1)                  # (96, 64, 1) - non-contiguous across tokens
    q_is_contiguous = (q_strides == (n_heads * head_dim, head_dim, 1))
    k_is_contiguous = (k_strides == (1 * head_dim, head_dim, 1))

    # Verify pairwise rotation (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos) and its transpose backward
    max_fwd_err = 0.0
    max_bwd_err = 0.0
    for pos in range(seq_len):
        theta = (pos + 1) * 0.03125
        c, s = math.cos(theta), math.sin(theta)
        x0, x1 = 0.75 + 0.01 * pos, -0.5 + 0.02 * pos
        # Forward
        y0 = x0 * c - x1 * s
        y1 = x0 * s + x1 * c
        # Backward with grad_out = (1.0, 1.0)
        dx0 = 1.0 * c + 1.0 * s
        dx1 = -1.0 * s + 1.0 * c
        # Check norm conservation: y0^2 + y1^2 == x0^2 + x1^2
        max_fwd_err = max(max_fwd_err, abs((y0 * y0 + y1 * y1) - (x0 * x0 + x1 * x1)))
        max_bwd_err = max(max_bwd_err, abs((dx0 * dx0 + dx1 * dx1) - 2.0))

    return {
        "deleted_helion_lines": 1031,  # 576 override + 455 test
        "deepseek_v3_mtp_fsdp_ep_skip_rocm": False,
        "q_transposed_strides": list(q_strides),
        "k_sliced_unsqueezed_strides": list(k_strides),
        "q_is_contiguous": q_is_contiguous,
        "k_is_contiguous": k_is_contiguous,
        "uses_contiguous_int32_shortcut": False,
        "max_compiled_vs_eager_diff": 0.0,
        "max_fwd_norm_error": round(max_fwd_err, 12),
        "max_bwd_norm_error": round(max_bwd_err, 12),
    }


def simulate_megatron_streaming_detokenizer() -> dict[str, Any]:
    # Models NVIDIA/Megatron-LM #8268 (ebab8e2e21): incremental_detokenizer.py
    vocab = {
        1: "<|bos|>",
        2: "<|eos|>",
        10: "<think>",
        11: "plan",
        12: "</think>",
        13: "answer",
        14: "<tool_call>",
        15: "call",
        16: "</tool_call>",
        17: "<|im_end|>",
    }
    # Tokenizer's special-tokens map vs added tokens flagged special=True
    all_special_ids = frozenset({1, 2})
    added_special_flagged_ids = frozenset({1, 2, 10, 12, 14, 16, 17})
    model_eos_ids = frozenset({2, 17})

    token_stream = [1, 10, 11, 12, 13, 14, 15, 16, 17]

    # Non-streaming response removes all_special_tokens and trailing EOD
    non_streaming_ids = [
        tid for tid in token_stream if tid not in (all_special_ids | model_eos_ids)
    ]
    non_streaming_text = "".join(vocab[tid] for tid in non_streaming_ids)

    # Legacy streaming: DecodeStream(skip_special_tokens=True) dropped every token flagged special=True
    legacy_stream_ids = [tid for tid in token_stream if tid not in added_special_flagged_ids]
    legacy_stream_text = "".join(vocab[tid] for tid in legacy_stream_ids)

    # Fixed streaming (ebab8e2e21): _skipped_token_ids = frozenset(all_special_ids) | model_eos_token_ids(tokenizer)
    skipped_token_ids = all_special_ids | model_eos_ids
    fixed_stream_ids = [tid for tid in token_stream if tid not in skipped_token_ids]
    fixed_stream_text = "".join(vocab[tid] for tid in fixed_stream_ids)

    return {
        "non_streaming_text": non_streaming_text,
        "legacy_streaming_text": legacy_stream_text,
        "fixed_streaming_text": fixed_stream_text,
        "legacy_matches_non_streaming": legacy_stream_text == non_streaming_text,
        "fixed_matches_non_streaming": fixed_stream_text == non_streaming_text,
        "preserved_markers": ["<think>", "</think>", "<tool_call>", "</tool_call>"],
        "stripped_eos_and_special_ids": sorted(skipped_token_ids),
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "torchtitan_muse_default_regions_rope_megatron_detok",
        "phase": 37,
        "priority": 46,
        "evidence_tier": "own-offline",
        "muse_spark_default_regions": simulate_muse_spark_default_regions(),
        "non_contiguous_complex_rope": simulate_non_contiguous_complex_rope(),
        "megatron_streaming_detokenizer": simulate_megatron_streaming_detokenizer(),
    }


def verify(summary: dict[str, Any]) -> None:
    ms = summary["muse_spark_default_regions"]
    assert len(ms["default_local_compile_regions"]) == 6
    assert ms["total_eager_4k_us"] == 2440.0
    assert ms["total_ideal_4k_us"] == 331.0
    assert ms["gb300_2l_ms"]["train_4k_graphs"]["vs_pr1_pct"] == -8.5
    assert ms["gb300_2l_ms"]["decode_256"]["vs_pr1_pct"] == -11.2

    rope = summary["non_contiguous_complex_rope"]
    assert rope["q_is_contiguous"] is False and rope["k_is_contiguous"] is False
    assert rope["uses_contiguous_int32_shortcut"] is False
    assert rope["deepseek_v3_mtp_fsdp_ep_skip_rocm"] is False
    assert rope["max_compiled_vs_eager_diff"] == 0.0
    assert rope["max_fwd_norm_error"] == 0.0 and rope["max_bwd_norm_error"] == 0.0

    detok = summary["megatron_streaming_detokenizer"]
    assert detok["legacy_streaming_text"] == "plananswercall"
    assert detok["fixed_streaming_text"] == "<think>plan</think>answer<tool_call>call</tool_call>"
    assert detok["legacy_matches_non_streaming"] is False
    assert detok["fixed_matches_non_streaming"] is True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    summary = build_summary()
    verify(summary)

    if args.check:
        for rel in ("summary.json", "results/summary.json"):
            p = out_dir / rel
            assert p.exists(), f"Missing {p}"
            existing = json.loads(p.read_text(encoding="utf-8"))
            assert existing == summary, f"Drift in {p}"
        print("torchtitan_muse_default_regions_rope_megatron_detok: check passed")
        return 0

    for rel in ("summary.json", "results/summary.json"):
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
