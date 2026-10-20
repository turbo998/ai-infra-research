from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def analyze_torchtitan_muse_spark_compile() -> dict[str, Any]:
    """Model TorchTitan PR #5434 (e923bfc7) & PR #5458 (08b4c0eb) Muse Spark compile regions."""
    # PR #5458: 4 call shapes (fp32 stream w/ weight, bf16 w/o weight, 4096-wide expert norm, final norm)
    # x 2 grad modes (grad / no_grad) + dynamic T specialization -> 12 graphs across train & inference.
    # Default local_compile recompile_limit is 8, so limit is raised to 16.
    default_recompile_limit = 8
    configured_recompile_limit = 16
    observed_rmsnorm_graphs = 12
    observed_attn_region_graphs = 4

    # PR #5434 64-wide per-head norm glue microbenchmarks at T=4096 (1x GB300, microseconds):
    qk_fwd_eager_us = 403.0
    qk_fwd_ideal_us = 21.0
    qk_fwd_bwd_glue_eager_us = 794.0
    qk_fwd_bwd_glue_ideal_us = 42.0
    out_fwd_bwd_glue_eager_us = 814.0
    out_fwd_bwd_glue_ideal_us = 66.0

    # End-to-end 2-layer Muse Spark Open 816B-A42B latency on 1x GB300 (ms, CUDA graphs):
    ab_table = {
        "train_4k_graphs_ms": {"main": 42.13, "pr2_base_regions": 35.94, "pr3_attn_glue": 33.35, "pr4_rmsnorm": 33.17},
        "train_16k_graphs_ms": {"main": 171.41, "pr2_base_regions": 147.88, "pr3_attn_glue": 138.02, "pr4_rmsnorm": 135.82},
        "prefill_2k_ms": {"main": 7.64, "pr2_base_regions": 6.45, "pr3_attn_glue": 5.71, "pr4_rmsnorm": 5.67},
        "prefill_16k_ms": {"main": 59.31, "pr2_base_regions": 50.65, "pr3_attn_glue": 45.18, "pr4_rmsnorm": 45.17},
        "decode_1_ms": {"main": 1.07, "pr2_base_regions": 0.98, "pr3_attn_glue": 0.96, "pr4_rmsnorm": 0.92},
        "decode_256_ms": {"main": 1.55, "pr2_base_regions": 1.32, "pr3_attn_glue": 1.23, "pr4_rmsnorm": 1.20},
    }

    return {
        "default_recompile_limit": default_recompile_limit,
        "configured_recompile_limit": configured_recompile_limit,
        "observed_rmsnorm_graphs": observed_rmsnorm_graphs,
        "observed_attn_region_graphs": observed_attn_region_graphs,
        "would_exceed_default_limit": observed_rmsnorm_graphs > default_recompile_limit,
        "within_configured_limit": observed_rmsnorm_graphs <= configured_recompile_limit,
        "qk_fwd_bandwidth_efficiency_eager": round(qk_fwd_ideal_us / qk_fwd_eager_us, 6),
        "total_attn_glue_eager_us_per_layer_4k": qk_fwd_bwd_glue_eager_us + out_fwd_bwd_glue_eager_us,
        "total_attn_glue_ideal_us_per_layer_4k": qk_fwd_bwd_glue_ideal_us + out_fwd_bwd_glue_ideal_us,
        "bf16_max_rel_error_vs_fp64": {
            "qk_norm_output_eager": 2.4e-3,
            "qk_norm_output_compiled": 1.7e-3,
            "out_norm_gate_bwd_eager": 4.3e-3,
            "out_norm_gate_bwd_compiled": 2.3e-3,
        },
        "gb300_ab_latencies_ms": ab_table,
        "stack_speedup_pct": {
            k: round((1.0 - v["pr4_rmsnorm"] / v["main"]) * 100.0, 2)
            for k, v in ab_table.items()
        },
    }


def _log_softmax(row: list[float]) -> list[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    lse = m + math.log(sum(exps))
    return [x - lse for x in row]


def _processed_row(row: list[float], temperature: float, top_k: int) -> list[float]:
    scaled = [x / max(temperature, 1e-4) for x in row]
    if 0 < top_k < len(scaled):
        threshold = sorted(scaled, reverse=True)[top_k - 1]
        masked = [x if x >= threshold else float("-inf") for x in scaled]
    else:
        masked = scaled
    finite_vals = [x for x in masked if math.isfinite(x)]
    m = max(finite_vals)
    lse = m + math.log(sum(math.exp(x - m) for x in finite_vals))
    return [x - lse if math.isfinite(x) else float("-inf") for x in masked]


def analyze_megatron_processed_logprobs(partial_chunk: bool = True) -> dict[str, Any]:
    """Model Megatron-LM PR #8162 (4f86e2c1) prompt vs sampled log-probs under processed_logprobs."""
    # Two active requests with query lengths [3, 2], vocab_size = 4
    query_lengths = [3, 2]
    sampling_params = [
        {"temperature": 0.5, "top_k": 2},
        {"temperature": 0.8, "top_k": 1},
    ]
    logits = [
        [2.0, 1.0, 0.5, -1.0],  # req 0 prompt row 0
        [1.5, 2.5, 0.2, -0.5],  # req 0 prompt row 1
        [3.0, 2.0, 1.0, 0.0],   # req 0 final row 2 (sampled token)
        [0.5, 1.5, 2.5, 3.5],   # req 1 prompt row 0
        [1.0, 4.0, 2.0, 0.0],   # req 1 final row 4 (next prompt token if partial_chunk else sampled)
    ]
    # Selected target tokens at each row (prompt next-tokens + sampled tokens at cumsum-1)
    # Deliberately pick token 3 (outside top-k) on row 0 and row 2 to test prompt rawness vs fallback
    target_tokens = [3, 2, 3, 0, 2]

    cumsum = []
    running = 0
    for q in query_lengths:
        running += q
        cumsum.append(running - 1)
    new_token_idx = cumsum  # [2, 4]

    raw_table = [_log_softmax(r) for r in logits]

    # Legacy bug: applied processed_log_probs to ALL rows via repeat_interleave(active_query_lengths)
    legacy_table = []
    for req_idx, q_len in enumerate(query_lengths):
        sp = sampling_params[req_idx]
        start = sum(query_lengths[:req_idx])
        for r in logits[start : start + q_len]:
            legacy_table.append(_processed_row(r, sp["temperature"], sp["top_k"]))

    # Fixed PR #8162:
    # 1) All rows start as raw log_softmax.
    # 2) Only sampled_idx = new_token_idx[:num_sampled] get processed_log_probs.
    # 3) If partial_chunk is True on last request, num_sampled = n_active - 1.
    n_active = len(query_lengths)
    num_sampled = n_active - 1 if partial_chunk else n_active
    sampled_idx = new_token_idx[:num_sampled]

    fixed_table = [list(r) for r in raw_table]
    for i, row_idx in enumerate(sampled_idx):
        sp = sampling_params[i]
        fixed_table[row_idx] = _processed_row(logits[row_idx], sp["temperature"], sp["top_k"])

    fixed_selected = []
    for row_idx, tok in enumerate(target_tokens):
        lp = fixed_table[row_idx][tok]
        if row_idx in sampled_idx and not math.isfinite(lp):
            req_i = sampled_idx.index(row_idx)
            tempered = _log_softmax([x / sampling_params[req_i]["temperature"] for x in logits[row_idx]])
            lp = tempered[tok]
        fixed_selected.append(round(lp, 6))

    legacy_prompt_row0_finite_count = sum(1 for x in legacy_table[0] if math.isfinite(x))
    fixed_prompt_row0_finite_count = sum(1 for x in fixed_table[0] if math.isfinite(x))
    fixed_partial_chunk_final_row_finite_count = sum(1 for x in fixed_table[4] if math.isfinite(x))

    return {
        "query_lengths": query_lengths,
        "new_token_idx": new_token_idx,
        "partial_chunk": partial_chunk,
        "num_sampled": num_sampled,
        "sampled_idx": sampled_idx,
        "legacy_prompt_row0_finite_support": legacy_prompt_row0_finite_count,
        "fixed_prompt_row0_finite_support": fixed_prompt_row0_finite_count,
        "fixed_sampled_row2_finite_support": sum(1 for x in fixed_table[2] if math.isfinite(x)),
        "fixed_partial_chunk_final_row4_finite_support": fixed_partial_chunk_final_row_finite_count,
        "fixed_selected_log_probs": fixed_selected,
        "all_selected_log_probs_finite": all(math.isfinite(x) for x in fixed_selected),
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "torchtitan_muse_spark_compile_megatron_logprobs",
        "torchtitan_muse_spark": analyze_torchtitan_muse_spark_compile(),
        "megatron_logprobs_partial_chunk": analyze_megatron_processed_logprobs(partial_chunk=True),
        "megatron_logprobs_full_prefill": analyze_megatron_processed_logprobs(partial_chunk=False),
    }


def verify_summary(summary: dict[str, Any]) -> None:
    tt = summary["torchtitan_muse_spark"]
    assert tt["would_exceed_default_limit"] is True
    assert tt["within_configured_limit"] is True
    assert tt["total_attn_glue_eager_us_per_layer_4k"] == 1608.0
    assert tt["total_attn_glue_ideal_us_per_layer_4k"] == 108.0
    assert tt["stack_speedup_pct"]["train_4k_graphs_ms"] == 21.27
    assert tt["stack_speedup_pct"]["prefill_16k_ms"] == 23.84

    mp_partial = summary["megatron_logprobs_partial_chunk"]
    assert mp_partial["sampled_idx"] == [2]
    assert mp_partial["legacy_prompt_row0_finite_support"] == 2
    assert mp_partial["fixed_prompt_row0_finite_support"] == 4
    assert mp_partial["fixed_sampled_row2_finite_support"] == 2
    assert mp_partial["fixed_partial_chunk_final_row4_finite_support"] == 4
    assert mp_partial["all_selected_log_probs_finite"] is True

    mp_full = summary["megatron_logprobs_full_prefill"]
    assert mp_full["sampled_idx"] == [2, 4]
    assert mp_full["fixed_partial_chunk_final_row4_finite_support"] == 1
    assert mp_full["all_selected_log_probs_finite"] is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    verify_summary(summary)

    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    res_dir = Path(__file__).resolve().parent / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        print("OK: torchtitan_muse_spark_compile_megatron_logprobs verified")


if __name__ == "__main__":
    main()
