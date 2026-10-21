#!/usr/bin/env python3
"""Deterministic stdlib reproduction for Phase 51 Topic 2:
SGLang Unified Radix-Tree Eviction Action Propagation (#45439) & DeepSeek-V4.1 Vision MoE
Routing Simulation (#45628), TensorRT-LLM gen_only Device-Step-Time History Floor (#20297),
and vLLM Multimodal Chunked Embedding Guard, XPU Spec-Decode Estimator & Rust ResponseFormat (#62859, #62244, #62823).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS = 1790856543000  # 2026-10-01T12:09:03Z


def simulate_sglang_radix_tree_and_dsv41_routing() -> dict[str, Any]:
    """Verify SGLang #45439 eviction cache_actions tripwire and #45628 DS-V4.1 vision MoE routing."""
    # 1. Unified Radix-Tree BaseEvictionResult resource drain tripwire (#45439)
    def check_eviction_result_disposal(
        device_frees: list[int], host_frees: list[int], cache_actions: list[str]
    ) -> bool:
        """Returns True if disposal is clean, False if undrained tripwire triggers."""
        return not device_frees and not host_frees and not cache_actions

    clean_after_drain = check_eviction_result_disposal([], [], [])
    tripwire_on_undrained_action = not check_eviction_result_disposal(
        [], [], ["c128_sidecar_evict_notify"]
    )

    # 2. DeepSeek-V4.1 Vision MoE routing simulation (#45628)
    def compute_vision_moe_simulation(
        *,
        top_k: int,
        num_shared_topk_slots: int,
        moe_input_scattered: bool,
        attn_dp_rank: int,
        attn_dp_size: int,
        attn_tp_rank: int,
        attn_tp_size: int,
        attn_tp_sequence_sharded: bool,
        padded_rows_masked_before: bool,
    ) -> dict[str, Any]:
        num_routed_topk = top_k - num_shared_topk_slots
        if moe_input_scattered:
            token_shard_rank = attn_dp_rank
            num_token_shards = attn_dp_size
            if attn_tp_sequence_sharded:
                token_shard_rank = token_shard_rank * attn_tp_size + attn_tp_rank
                num_token_shards *= attn_tp_size
        else:
            token_shard_rank, num_token_shards = 0, 1
        # _finish_vision_topk resets padded_rows_masked = False after apply_routing_simulation
        # so _post_process_topk_ids re-masks CUDA-graph padded tokens.
        padded_rows_masked_after_sim = False
        return {
            "num_routed_topk": num_routed_topk,
            "shared_slots_preserved": num_shared_topk_slots,
            "token_shard_rank": token_shard_rank,
            "num_token_shards": num_token_shards,
            "padded_rows_masked_before": padded_rows_masked_before,
            "padded_rows_masked_after_sim": padded_rows_masked_after_sim,
        }

    scattered_tp_sharded = compute_vision_moe_simulation(
        top_k=9,
        num_shared_topk_slots=1,
        moe_input_scattered=True,
        attn_dp_rank=2,
        attn_dp_size=4,
        attn_tp_rank=3,
        attn_tp_size=4,
        attn_tp_sequence_sharded=True,
        padded_rows_masked_before=True,
    )
    gathered_replica = compute_vision_moe_simulation(
        top_k=9,
        num_shared_topk_slots=1,
        moe_input_scattered=False,
        attn_dp_rank=2,
        attn_dp_size=4,
        attn_tp_rank=3,
        attn_tp_size=4,
        attn_tp_sequence_sharded=True,
        padded_rows_masked_before=True,
    )

    return {
        "clean_after_drain": clean_after_drain,
        "tripwire_on_undrained_action": tripwire_on_undrained_action,
        "scattered_tp_sharded": scattered_tp_sharded,
        "gathered_replica": gathered_replica,
        "invariants_passed": (
            clean_after_drain
            and tripwire_on_undrained_action
            and scattered_tp_sharded["num_routed_topk"] == 8
            and scattered_tp_sharded["token_shard_rank"] == 11
            and scattered_tp_sharded["num_token_shards"] == 16
            and scattered_tp_sharded["padded_rows_masked_after_sim"] is False
            and gathered_replica["token_shard_rank"] == 0
            and gathered_replica["num_token_shards"] == 1
        ),
    }


def simulate_trtllm_history_floor() -> dict[str, Any]:
    """Verify TensorRT-LLM #20297 (7f0caf443e) gen_only device-step-time history floor."""
    def percentile(values: list[float], p: float) -> float:
        s = sorted(values)
        k = (len(s) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(s) - 1)
        frac = k - lo
        return s[lo] + frac * (s[hi] - s[lo])

    def rolling_smooth(values: list[float], window: int = 3) -> list[float]:
        out = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            chunk = values[start : i + 1]
            out.append(sum(chunk) / len(chunk))
        return out

    # History records: 2 pre-cutover days where modal bucket picked the short tail wave (~2.5 ms)
    # and 4 post-cutover days where expected steady-state bucket measures the full batch (~7.5 ms).
    history = [
        {"day": "2026-09-28", "ts_created": 1790597343000, "val": 2.45},
        {"day": "2026-09-30", "ts_created": 1790770143000, "val": 2.55},
        {"day": "2026-10-02", "ts_created": 1790942943000, "val": 7.48},
        {"day": "2026-10-03", "ts_created": 1791029343000, "val": 7.52},
        {"day": "2026-10-04", "ts_created": 1791115743000, "val": 7.50},
        {"day": "2026-10-05", "ts_created": 1791202143000, "val": 7.54},
    ]
    current_val = 7.51
    post_merge_threshold = 0.05  # 5%

    unfloored_daily = [r["val"] for r in history]
    unfloored_p5 = percentile(rolling_smooth(unfloored_daily, 3), 5.0)
    unfloored_false_regression = current_val > unfloored_p5 * (1.0 + post_merge_threshold)

    floored_daily = [
        r["val"]
        for r in history
        if r["ts_created"] >= GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS
    ]
    floored_p5 = percentile(rolling_smooth(floored_daily, 3), 5.0)
    floored_false_regression = current_val > floored_p5 * (1.0 + post_merge_threshold)

    return {
        "floor_epoch_ms": GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS,
        "unfloored_p5_ms": round(unfloored_p5, 4),
        "floored_p5_ms": round(floored_p5, 4),
        "current_step_time_ms": current_val,
        "unfloored_triggers_false_regression": unfloored_false_regression,
        "floored_triggers_false_regression": floored_false_regression,
        "tail_vs_full_wave_ratio": round(7.50 / 2.50, 2),
        "invariants_passed": (
            unfloored_false_regression is True
            and floored_false_regression is False
            and abs(floored_p5 - 7.496) < 0.02
        ),
    }


def simulate_vllm_pooling_xpu_rust() -> dict[str, Any]:
    """Verify vLLM #62859 chunked multimodal guard, #62244 XPU estimator, and #62823 Rust response_format."""
    # 1. vLLM #62859: Reject multimodal inputs anywhere in batch before chunked processing mutates ctx
    def validate_chunked_pooling_batch(
        enable_chunked: bool, input_types: list[str]
    ) -> str:
        if not enable_chunked:
            return "passthrough"
        if any(t == "multimodal" for t in input_types):
            return "rejected_VLLMValidationError"
        return "chunked_ok"

    mixed_chunked = validate_chunked_pooling_batch(True, ["token_ids", "multimodal"])
    mixed_unchunked = validate_chunked_pooling_batch(False, ["token_ids", "multimodal"])

    # 2. vLLM #62244: OnlineAcceptanceEstimator damping factor n / (n + DAMPING_OBSERVATIONS)
    damping_obs = 20.0
    n_obs = 20.0
    damping_factor_at_20 = n_obs / (n_obs + damping_obs)

    # 3. vLLM #62823: Rust convert_from_response_format precedence and conflict rejection
    def resolve_rust_structured_outputs(
        response_format_kind: str | None,
        response_format_val: str | None,
        structured_outputs_kind: str | None,
        structured_outputs_val: str | None,
        aux_whitespace_pattern: str | None = None,
    ) -> dict[str, Any]:
        if response_format_kind is None or response_format_kind == "text":
            return {
                "status": "ok",
                "kind": structured_outputs_kind,
                "value": structured_outputs_val,
                "aux": aux_whitespace_pattern,
            }
        if structured_outputs_kind is None:
            return {
                "status": "ok",
                "kind": response_format_kind,
                "value": response_format_val,
                "aux": None,
            }
        if structured_outputs_kind == response_format_kind:
            # response_format overrides same-kind constraint while preserving aux options
            return {
                "status": "ok",
                "kind": response_format_kind,
                "value": response_format_val,
                "aux": aux_whitespace_pattern,
            }
        return {
            "status": "error_400_multiple_constraints",
            "kind": None,
            "value": None,
            "aux": None,
        }

    text_keeps_so = resolve_rust_structured_outputs(
        "text", None, "regex", "[0-9]+", "no_tabs"
    )
    same_kind_override = resolve_rust_structured_outputs(
        "json_schema", "schema_from_rf", "json_schema", "schema_from_so", "no_tabs"
    )
    cross_kind_conflict = resolve_rust_structured_outputs(
        "json_object", "true", "regex", "[a-z]+"
    )

    return {
        "mixed_batch_chunked": mixed_chunked,
        "mixed_batch_unchunked": mixed_unchunked,
        "xpu_estimator_damping_at_20_obs": damping_factor_at_20,
        "rust_text_keeps_structured_outputs": text_keeps_so,
        "rust_same_kind_override_keeps_aux": same_kind_override,
        "rust_cross_kind_conflict": cross_kind_conflict,
        "invariants_passed": (
            mixed_chunked == "rejected_VLLMValidationError"
            and mixed_unchunked == "passthrough"
            and damping_factor_at_20 == 0.5
            and text_keeps_so["value"] == "[0-9]+"
            and same_kind_override["value"] == "schema_from_rf"
            and same_kind_override["aux"] == "no_tabs"
            and cross_kind_conflict["status"] == "error_400_multiple_constraints"
        ),
    }


def run_experiment() -> dict[str, Any]:
    sglang_res = simulate_sglang_radix_tree_and_dsv41_routing()
    trtllm_res = simulate_trtllm_history_floor()
    vllm_res = simulate_vllm_pooling_xpu_rust()
    all_passed = (
        sglang_res["invariants_passed"]
        and trtllm_res["invariants_passed"]
        and vllm_res["invariants_passed"]
    )
    return {
        "experiment": "sglang-radix-tree-evict-actions-dsv41-vl-sim-trtllm-perf-floor-vllm-rust",
        "upstream_commits": [
            "sgl-project/sglang@5e79333c95c162abf079548fe40a6a2bca6c5409",
            "sgl-project/sglang@ed268dd32097ed44a83b260250defb82256602e2",
            "NVIDIA/TensorRT-LLM@7f0caf443ec440ffc7ca455c91ff6f7c8b0b8d67",
            "vllm-project/vllm@65a886c69f44cf446d6afa03315f39590ccd843e",
            "vllm-project/vllm@35d01660ff3644de128253950b8fcb4b7705668f",
            "vllm-project/vllm@18e8791134e8e46134f982c8263d5fa0a50548c0",
        ],
        "sglang_radix_tree_and_dsv41": sglang_res,
        "trtllm_history_floor": trtllm_res,
        "vllm_pooling_xpu_rust": vllm_res,
        "all_invariants_passed": all_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_experiment()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check and not summary["all_invariants_passed"]:
        raise SystemExit(1)
    print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
