#!/usr/bin/env python3
"""CPU/offline analytical and discrete-event model for AI Infra Innovators v2:
Adaptive EAGLE-3 Speculative Decoding, NVFP4/FP8 Tensor & KV-Scale Swizzling, and Wafer/SRAM/Rack Pareto Frontier.

Covers exact primary-source algorithms from `sgl-project/sglang` (`4ec19ca27f08f7ce44bbe1da70ce964731940eb2`):
1. `python/sglang/srt/speculative/adaptive_spec_params.py:27-52, 143-348` (`AdaptiveStepSlot` & `AdaptiveSpeculativeParams`):
   Exact batch-size tiered candidate steps (`{"1": [1,3,5,7], "8": [0,1,3], "32": [0,1], "64": [0]}`), EMA smoothing,
   hysteresis (`down_hysteresis = -0.25`, `up_hysteresis = 0.0`), and roof-line verification why large batch sizes
   must disable drafting (`steps = 0`) when verify shifts from HBM-bandwidth-bound to Tensor-Core-compute-bound.
2. `python/sglang/srt/layers/quantization/fp4_utils.py:26-83` & `nvfp4_kv_cache.py:33-54`:
   Exact `128x4` swizzled block-scale factor sizing (`sf_rows = round_up(m, 128)`, `sf_cols = round_up(K // 16, 4)`)
   and TRT-LLM native 4-token-interleaved NVFP4 V-scale permutation (`nvfp4_v_scale_swizzle_indices`).
3. Wafer-Scale / SRAM-LPU / NVL72-Rack / Cloud-B200 Latency-vs-Cost Pareto Frontier across
   Groq LPU, Cerebras WSE-3, Fireworks/Together/Baseten/Modal on B200 NVFP4 + EAGLE-3, and GB200 NVL72.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input.json"
DEFAULT_JSON = ROOT / "results.json"
DEFAULT_CSV = ROOT / "results.csv"


def _round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _validate_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _validate_positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return val


def nvfp4_v_scale_swizzle_coord(token_offset: int, scale_idx: int, scale_dim: int) -> tuple[int, int]:
    """Exact port of `nvfp4_v_scale_swizzle_indices` (`nvfp4_kv_cache.py:33-53`)."""
    if scale_dim <= 0 or scale_dim % 4 != 0:
        raise ValueError(f"NVFP4 scale_dim must be positive and divisible by 4, got {scale_dim}.")
    if token_offset < 0 or scale_idx < 0 or scale_idx >= scale_dim:
        raise ValueError(f"Invalid token_offset={token_offset} or scale_idx={scale_idx} for scale_dim={scale_dim}")
    scale_group = scale_dim // 4
    swizzled_token = (token_offset // 4) * 4 + scale_idx // scale_group
    swizzled_scale = (scale_idx % scale_group) * 4 + token_offset % 4
    return swizzled_token, swizzled_scale


def compute_fp4_quantized_shapes(
    m: int,
    k: int,
    sf_vec_size: int = 16,
    is_sf_swizzled_layout: bool = True,
    is_sf_8x4_layout: bool = False,
) -> dict[str, Any]:
    """Exact port of `_flashinfer_fp4_quantize_fake` (`fp4_utils.py:51-82`)."""
    if m <= 0 or k <= 0 or k % sf_vec_size != 0:
        raise ValueError(f"Invalid m={m}, k={k}, sf_vec_size={sf_vec_size}")
    xq_bytes = m * (k // 2)
    linear_sf_bytes = m * (k // sf_vec_size)
    if is_sf_swizzled_layout:
        row_size = 8 if is_sf_8x4_layout else 128
        sf_rows = _round_up(m, row_size)
        sf_cols = _round_up(k // sf_vec_size, 4)
    else:
        sf_rows = m
        sf_cols = k // sf_vec_size
    swizzled_sf_bytes = sf_rows * sf_cols
    total_fp4_bytes = xq_bytes + swizzled_sf_bytes
    bf16_bytes = m * k * 2
    fp8_bytes = m * k + m * (k // 128) * 4
    return {
        "m": m,
        "k": k,
        "xq_bytes": xq_bytes,
        "linear_sf_bytes": linear_sf_bytes,
        "swizzled_sf_rows": sf_rows,
        "swizzled_sf_cols": sf_cols,
        "swizzled_sf_bytes": swizzled_sf_bytes,
        "sf_padding_overhead_pct": round((swizzled_sf_bytes - linear_sf_bytes) / float(linear_sf_bytes) * 100.0, 2),
        "total_fp4_bytes": total_fp4_bytes,
        "effective_bits_per_elem": round((total_fp4_bytes * 8.0) / float(m * k), 3),
        "compression_vs_bf16_x": round(bf16_bytes / float(total_fp4_bytes), 2),
        "compression_vs_fp8_x": round(fp8_bytes / float(total_fp4_bytes), 2),
    }


class AdaptiveStepSimulator:
    """Exact port of `AdaptiveStepSlot` (`adaptive_spec_params.py:143-263`)."""

    def __init__(self, initial_steps: int, cfg: dict[str, Any]) -> None:
        candidates = sorted(set(int(x) for x in cfg["candidate_steps"]))
        if not candidates:
            raise ValueError("candidate_steps must be non-empty")
        self.candidate_steps = candidates
        self.ema_alpha = float(cfg.get("ema_alpha", 0.2))
        self.update_interval = int(cfg.get("update_interval", 5))
        self.warmup_batches = int(cfg.get("warmup_batches", 10))
        self.down_hysteresis = float(cfg.get("down_hysteresis", -0.25))
        self.up_hysteresis = float(cfg.get("up_hysteresis", 0.0))
        self.ceiling_coeff = float(cfg.get("ceiling_coeff", 0))

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]
        self.ema_accept_len = float(self.current_steps - 1)
        self.batch_count = 0

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        if not num_correct_drafts_per_req:
            return False
        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / float(len(num_correct_drafts_per_req))
            self.ema_accept_len = (1.0 - self.ema_alpha) * self.ema_accept_len + self.ema_alpha * batch_avg

        self.batch_count += 1
        if self.batch_count <= self.warmup_batches:
            return False
        if (self.batch_count - self.warmup_batches) % self.update_interval != 0:
            return False
        return self._recompute_params()

    def _recompute_params(self) -> bool:
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)
        old_idx = current_idx

        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                self.ema_accept_len = float(target - 1)
            self.current_steps = target
            return target != old_steps

        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        moved_down = current_idx < old_idx
        if not moved_down:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis
                if self.ema_accept_len > rise_threshold:
                    current_idx += 1
                else:
                    break

        target = self.candidate_steps[current_idx]
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]

        self.current_steps = target
        return target != old_steps


def evaluate_adaptive_eagle3_case(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate EAGLE-3 step latency/throughput vs Fixed Steps vs SGLang `AdaptiveStepSlot`."""
    batch_size = _validate_positive_int("batch_size", case.get("batch_size"))
    candidate_steps = case.get("candidate_steps")
    if not isinstance(candidate_steps, list) or not candidate_steps:
        raise ValueError("candidate_steps must be a non-empty list")
    initial_steps = _validate_non_negative_int("initial_steps", case.get("initial_steps", 3))
    fixed_naive_steps = _validate_positive_int("fixed_naive_steps", case.get("fixed_naive_steps", 5))
    token_accept_rate = _validate_positive_float("token_accept_rate", case.get("token_accept_rate"))
    if token_accept_rate > 1.0:
        raise ValueError("token_accept_rate must be <= 1.0")

    target_weight_gb = _validate_positive_float("target_weight_gb", case.get("target_weight_gb", 68.0))
    draft_weight_gb = _validate_positive_float("draft_weight_gb", case.get("draft_weight_gb", 1.2))
    kv_gb_per_req = _validate_positive_float("kv_gb_per_req", case.get("kv_gb_per_req", 0.25))
    hbm_bw_gb_s = _validate_positive_float("hbm_bw_gb_s", case.get("hbm_bw_gb_s", 7200.0))
    peak_tflops = _validate_positive_float("peak_tflops", case.get("peak_tflops", 2250.0))
    active_params_b = _validate_positive_float("active_params_b", case.get("active_params_b", 37.0))
    draft_params_b = _validate_positive_float("draft_params_b", case.get("draft_params_b", 0.8))
    kernel_launch_overhead_ms = _validate_positive_float(
        "kernel_launch_overhead_ms", case.get("kernel_launch_overhead_ms", 0.08)
    )

    def step_metrics(num_steps: int) -> tuple[float, float, float]:
        # Expected accepted draft tokens for chain length `num_steps` with per-step acceptance `alpha`:
        if num_steps == 0:
            accepted_drafts = 0.0
        elif abs(1.0 - token_accept_rate) < 1e-9:
            accepted_drafts = float(num_steps)
        else:
            accepted_drafts = token_accept_rate * (1.0 - (token_accept_rate ** num_steps)) / (1.0 - token_accept_rate)
        emitted_tokens = 1.0 + accepted_drafts

        # Verify stage processes `batch_size * (num_steps + 1)` tokens in one target forward pass:
        verify_tokens = batch_size * (num_steps + 1)
        verify_mem_ms = ((target_weight_gb + batch_size * kv_gb_per_req) / hbm_bw_gb_s) * 1000.0
        verify_comp_ms = ((2.0 * active_params_b * 1e9 * verify_tokens) / (peak_tflops * 1e12)) * 1000.0
        verify_ms = max(verify_mem_ms, verify_comp_ms) + kernel_launch_overhead_ms

        # Draft stage runs `num_steps` autoregressive steps + 1 `draft_extend` KV catch-up pass:
        if num_steps == 0:
            draft_ms = 0.0
        else:
            per_draft_mem_ms = ((draft_weight_gb + batch_size * 0.02 * kv_gb_per_req) / hbm_bw_gb_s) * 1000.0
            per_draft_comp_ms = ((2.0 * draft_params_b * 1e9 * batch_size) / (peak_tflops * 1e12)) * 1000.0
            draft_step_ms = max(per_draft_mem_ms, per_draft_comp_ms) + kernel_launch_overhead_ms
            extend_tokens = batch_size * (num_steps + 1)
            extend_comp_ms = ((2.0 * draft_params_b * 1e9 * extend_tokens) / (peak_tflops * 1e12)) * 1000.0
            extend_ms = max(per_draft_mem_ms, extend_comp_ms) + kernel_launch_overhead_ms
            draft_ms = num_steps * draft_step_ms + extend_ms

        total_cycle_ms = verify_ms + draft_ms
        tpot_ms = total_cycle_ms / emitted_tokens
        cluster_tok_s = (batch_size * emitted_tokens) / (total_cycle_ms / 1000.0)
        return round(accepted_drafts, 3), round(tpot_ms, 3), round(cluster_tok_s, 1)

    sim = AdaptiveStepSimulator(
        initial_steps=initial_steps,
        cfg={
            "candidate_steps": candidate_steps,
            "down_hysteresis": case.get("down_hysteresis", -0.25),
            "up_hysteresis": case.get("up_hysteresis", 0.0),
            "warmup_batches": 10,
            "update_interval": 5,
        },
    )
    for _ in range(30):
        cur = sim.current_steps
        acc, _, _ = step_metrics(cur)
        sim.update([int(round(acc))] * batch_size)

    converged_steps = sim.current_steps
    _, nospec_tpot_ms, nospec_tok_s = step_metrics(0)
    _, fixed_tpot_ms, fixed_tok_s = step_metrics(fixed_naive_steps)
    conv_acc, adaptive_tpot_ms, adaptive_tok_s = step_metrics(converged_steps)

    return {
        "case_id": str(case["case_id"]),
        "batch_size": batch_size,
        "token_accept_rate": token_accept_rate,
        "candidate_steps": candidate_steps,
        "converged_adaptive_steps": converged_steps,
        "expected_accepted_drafts": conv_acc,
        "nospec_tpot_ms": nospec_tpot_ms,
        "fixed_step5_tpot_ms": fixed_tpot_ms,
        "adaptive_tpot_ms": adaptive_tpot_ms,
        "nospec_throughput_tok_s": nospec_tok_s,
        "fixed_step5_throughput_tok_s": fixed_tok_s,
        "adaptive_throughput_tok_s": adaptive_tok_s,
        "adaptive_vs_fixed_speedup_x": round(fixed_tpot_ms / adaptive_tpot_ms, 2),
        "adaptive_vs_nospec_speedup_x": round(nospec_tpot_ms / adaptive_tpot_ms, 2),
    }


def evaluate_pareto_platform(case: dict[str, Any]) -> dict[str, Any]:
    """Compare Wafer-Scale / SRAM-LPU / NVL72 / B200-Cloud on Latency-vs-Cost Pareto Frontier."""
    platform = str(case["platform"])
    tier = str(case["tier"])
    active_weight_gb = _validate_positive_float("active_weight_gb", case.get("active_weight_gb"))
    effective_mem_bw_tb_s = _validate_positive_float("effective_mem_bw_tb_s", case.get("effective_mem_bw_tb_s"))
    interconnect_overhead_ms = _validate_positive_float(
        "interconnect_overhead_ms", case.get("interconnect_overhead_ms")
    )
    spec_accept_multiplier = _validate_positive_float("spec_accept_multiplier", case.get("spec_accept_multiplier", 1.0))
    spec_step_overhead_factor = _validate_positive_float(
        "spec_step_overhead_factor", case.get("spec_step_overhead_factor", 1.0)
    )
    max_batch_under_slo = _validate_positive_int("max_batch_under_slo", case.get("max_batch_under_slo"))
    system_hourly_cost_usd = _validate_positive_float("system_hourly_cost_usd", case.get("system_hourly_cost_usd"))
    max_context_k = _validate_positive_int("max_context_k", case.get("max_context_k"))

    raw_step_ms = ((active_weight_gb / (effective_mem_bw_tb_s * 1000.0)) * 1000.0) + interconnect_overhead_ms
    effective_cycle_ms = raw_step_ms * spec_step_overhead_factor
    single_user_tpot_ms = effective_cycle_ms / spec_accept_multiplier
    single_user_tok_s = 1000.0 / single_user_tpot_ms

    # At `max_batch_under_slo`, compute interference increases cycle time by ~25%
    slo_cycle_ms = effective_cycle_ms * 1.25
    system_tok_s_at_slo = (max_batch_under_slo * spec_accept_multiplier) / (slo_cycle_ms / 1000.0)
    cost_per_1m_output_tokens_usd = (system_hourly_cost_usd / 3600.0) * (1e6 / system_tok_s_at_slo)

    return {
        "case_id": str(case["case_id"]),
        "platform": platform,
        "tier": tier,
        "single_user_tpot_ms": round(single_user_tpot_ms, 3),
        "single_user_tok_s": round(single_user_tok_s, 1),
        "max_batch_under_slo": max_batch_under_slo,
        "system_tok_s_at_slo": round(system_tok_s_at_slo, 1),
        "cost_per_1m_output_tokens_usd": round(cost_per_1m_output_tokens_usd, 4),
        "max_context_k": max_context_k,
    }


def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise ValueError("Fixture schema_version must be 1")

    adaptive_cases = fixture.get("adaptive_eagle3_cases")
    fp4_cases = fixture.get("fp4_quant_and_swizzle_cases")
    pareto_cases = fixture.get("innovator_pareto_cases")
    if not isinstance(adaptive_cases, list) or not adaptive_cases:
        raise ValueError("adaptive_eagle3_cases must be a non-empty list")
    if not isinstance(fp4_cases, list) or not fp4_cases:
        raise ValueError("fp4_quant_and_swizzle_cases must be a non-empty list")
    if not isinstance(pareto_cases, list) or not pareto_cases:
        raise ValueError("innovator_pareto_cases must be a non-empty list")

    fp4_results: list[dict[str, Any]] = []
    for c in fp4_cases:
        res = compute_fp4_quantized_shapes(
            m=_validate_positive_int("m", c.get("m")),
            k=_validate_positive_int("k", c.get("k")),
            sf_vec_size=_validate_positive_int("sf_vec_size", c.get("sf_vec_size", 16)),
            is_sf_swizzled_layout=bool(c.get("is_sf_swizzled_layout", True)),
            is_sf_8x4_layout=bool(c.get("is_sf_8x4_layout", False)),
        )
        sw_tok, sw_sc = nvfp4_v_scale_swizzle_coord(
            token_offset=_validate_non_negative_int("sample_token_offset", c.get("sample_token_offset", 3)),
            scale_idx=_validate_non_negative_int("sample_scale_idx", c.get("sample_scale_idx", 5)),
            scale_dim=_validate_positive_int("kv_scale_dim", c.get("kv_scale_dim", 8)),
        )
        res["case_id"] = str(c["case_id"])
        res["v_scale_swizzled_token"] = sw_tok
        res["v_scale_swizzled_scale"] = sw_sc
        fp4_results.append(res)

    return {
        "schema_version": 1,
        "experiment_id": "innovator_pareto_v2",
        "evidence_type": "own-offline",
        "adaptive_eagle3": [evaluate_adaptive_eagle3_case(c) for c in adaptive_cases],
        "fp4_quant_and_swizzle": fp4_results,
        "innovator_pareto": [evaluate_pareto_platform(c) for c in pareto_cases],
    }


def _to_csv_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results["adaptive_eagle3"]:
        rows.append(
            {
                "section": "adaptive_eagle3",
                "case_id": item["case_id"],
                "primary_metric": item["adaptive_tpot_ms"],
                "secondary_metric": item["fixed_step5_tpot_ms"],
                "delta_or_ratio": item["adaptive_vs_fixed_speedup_x"],
                "note": f"steps={item['converged_adaptive_steps']}",
            }
        )
    for item in results["fp4_quant_and_swizzle"]:
        rows.append(
            {
                "section": "fp4_quant_and_swizzle",
                "case_id": item["case_id"],
                "primary_metric": item["effective_bits_per_elem"],
                "secondary_metric": item["sf_padding_overhead_pct"],
                "delta_or_ratio": item["compression_vs_bf16_x"],
                "note": f"swizzle=({item['v_scale_swizzled_token']},{item['v_scale_swizzled_scale']})",
            }
        )
    for item in results["innovator_pareto"]:
        rows.append(
            {
                "section": "innovator_pareto",
                "case_id": item["case_id"],
                "primary_metric": item["single_user_tok_s"],
                "secondary_metric": item["cost_per_1m_output_tokens_usd"],
                "delta_or_ratio": item["system_tok_s_at_slo"],
                "note": item["tier"],
            }
        )
    return rows


def _format_csv(rows: list[dict[str, Any]]) -> str:
    fieldnames = ["section", "case_id", "primary_metric", "secondary_metric", "delta_or_ratio", "note"]
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true", help="Verify committed outputs match without writing")
    args = parser.parse_args(argv)

    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    results = evaluate_fixture(fixture)
    json_text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    csv_text = _format_csv(_to_csv_rows(results))

    if args.check:
        if not args.json_out.exists() or not args.csv_out.exists():
            raise SystemExit("Committed results.json or results.csv missing")
        if args.json_out.read_text(encoding="utf-8") != json_text:
            raise SystemExit("results.json does not match deterministic evaluation")
        if args.csv_out.read_text(encoding="utf-8") != csv_text:
            raise SystemExit("results.csv does not match deterministic evaluation")
        print("own-offline innovator_pareto_v2 outputs verified")
        return 0

    args.json_out.write_text(json_text, encoding="utf-8")
    args.csv_out.write_text(csv_text, encoding="utf-8")
    print("own-offline innovator_pareto_v2 outputs generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
