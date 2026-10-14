"""Evaluate Fireworks KV-outer byte models, Together Sequoia tree Roofline, and Groq/Cerebras SRAM scaling."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__:
    from ..capacity_roofline.budget import integer
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.capacity_roofline.budget import integer


HERE = Path(__file__).resolve().parent
LIMITATIONS = [
    "own-offline synthetic arithmetic; no Blackwell GPU, H100, Groq LPU, or Cerebras WSE hardware execution",
    "kvouter byte formulas reproduce the analytical fio1x/fioAct/cio/ioAlt models defined in fw-ai/minimax-kernels",
    "speculative tree Roofline models ideal weight-read vs GEMM-compute bounds without KV-cache capacity contention",
    "spatial SRAM capacity calculation is a first-order lower bound on unit count and weight streaming time",
]


def _ascii_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or not value.replace("_", "").isalnum():
        raise ValueError(f"{name}: expected non-empty ASCII alphanumeric/underscore label")
    return value


def evaluate_kvouter_case(
    *,
    case_id: str,
    tq: int,
    hq: int,
    hkv: int,
    head_dim: int,
    block_size: int,
    topk: int,
    distinct_blocks: int,
    input_bytes: int,
    partial_bytes: int,
    out_bytes: int,
) -> dict:
    """Reproduce Fireworks M3 KV-outer fio1x, fioAct, cio, and no-reuse ioAlt byte models."""
    label = _ascii_label(case_id, "case_id")
    q_tokens = integer(tq, "tq")
    q_heads = integer(hq, "hq")
    kv_heads = integer(hkv, "hkv")
    dim = integer(head_dim, "head_dim")
    blk_tokens = integer(block_size, "block_size")
    k_sel = integer(topk, "topk")
    num_sel = integer(distinct_blocks, "distinct_blocks")
    in_b = integer(input_bytes, "input_bytes")
    part_b = integer(partial_bytes, "partial_bytes")
    out_b = integer(out_bytes, "out_bytes")

    if q_heads % kv_heads != 0:
        raise ValueError("hq must be divisible by hkv")
    total_pairs = q_tokens * k_sel
    if num_sel > total_pairs:
        raise ValueError("distinct_blocks cannot exceed tq * topk")

    qhead_group = q_heads // kv_heads
    rows = total_pairs * qhead_group
    reuse_factor_m = total_pairs / num_sel
    kv_block_bytes = 2 * blk_tokens * dim * in_b

    q_once_bytes = q_tokens * q_heads * dim * in_b
    q_restream_bytes = rows * dim * in_b
    kv_unique_bytes = kv_heads * num_sel * kv_block_bytes
    kv_no_reuse_bytes = kv_heads * total_pairs * kv_block_bytes
    # Each partial row writes `dim * partial_bytes` (o_partial) + 8 bytes (FP32 m_partial + FP32 l_partial)
    partials_write_bytes = kv_heads * rows * (dim * part_b + 8)
    inv_map_bytes = kv_heads * total_pairs * 4
    final_out_bytes = q_tokens * q_heads * dim * out_b

    fio1x_bytes = q_once_bytes + kv_unique_bytes + partials_write_bytes
    fio_act_bytes = q_restream_bytes + kv_unique_bytes + partials_write_bytes
    io_alt_no_reuse_bytes = q_once_bytes + kv_no_reuse_bytes + partials_write_bytes
    cio_bytes = partials_write_bytes + inv_map_bytes + final_out_bytes

    return {
        "case_id": label,
        "total_pairs_per_kv_head": total_pairs,
        "distinct_selected_blocks": num_sel,
        "reuse_factor_m": reuse_factor_m,
        "packed_gqa_rows_per_kv_head": rows,
        "kv_block_bytes": kv_block_bytes,
        "q_once_bytes": q_once_bytes,
        "q_restream_bytes": q_restream_bytes,
        "kv_unique_bytes": kv_unique_bytes,
        "kv_no_reuse_bytes": kv_no_reuse_bytes,
        "partials_write_bytes": partials_write_bytes,
        "fio1x_gb": round(fio1x_bytes / 1e9, 6),
        "fio_act_gb": round(fio_act_bytes / 1e9, 6),
        "io_alt_no_reuse_gb": round(io_alt_no_reuse_bytes / 1e9, 6),
        "cio_gb": round(cio_bytes / 1e9, 6),
        "kv_only_compression_ratio": kv_no_reuse_bytes / kv_unique_bytes,
        "total_forward_io_saving_ratio_vs_no_reuse": round(io_alt_no_reuse_bytes / fio_act_bytes, 6),
    }


def evaluate_speculative_tree_case(
    *,
    case_id: str,
    model_params: int,
    weight_bytes_per_param: int,
    tree_tokens_k: int,
    expected_accepted_tokens: float,
    draft_overhead_seconds: float,
    hbm_bytes_per_second: int,
    peak_flops_per_second: int,
) -> dict:
    """Evaluate Together Sequoia tree verification Roofline and net token throughput gain."""
    label = _ascii_label(case_id, "case_id")
    params = integer(model_params, "model_params")
    w_bytes = integer(weight_bytes_per_param, "weight_bytes_per_param")
    k_tokens = integer(tree_tokens_k, "tree_tokens_k")
    hbm_bw = integer(hbm_bytes_per_second, "hbm_bytes_per_second")
    peak_flops = integer(peak_flops_per_second, "peak_flops_per_second")
    if (
        not isinstance(expected_accepted_tokens, (int, float))
        or isinstance(expected_accepted_tokens, bool)
        or not math.isfinite(expected_accepted_tokens)
        or expected_accepted_tokens < 1.0
    ):
        raise ValueError("expected_accepted_tokens: expected finite number >= 1.0")
    if (
        not isinstance(draft_overhead_seconds, (int, float))
        or isinstance(draft_overhead_seconds, bool)
        or not math.isfinite(draft_overhead_seconds)
        or draft_overhead_seconds < 0.0
    ):
        raise ValueError("draft_overhead_seconds: expected finite number >= 0.0")

    weight_bytes_total = params * w_bytes
    verify_flops = 2 * params * k_tokens
    arithmetic_intensity = verify_flops / weight_bytes_total
    ridge_flops_per_byte = peak_flops / hbm_bw

    memory_time_s = weight_bytes_total / hbm_bw
    compute_time_s = verify_flops / peak_flops
    verify_step_time_s = max(memory_time_s, compute_time_s) + float(draft_overhead_seconds)
    b1_baseline_time_s = max(memory_time_s, (2 * params) / peak_flops)

    regime = "compute_bound" if compute_time_s > memory_time_s else "memory_bound"
    effective_tokens_per_second = float(expected_accepted_tokens) / verify_step_time_s
    baseline_tokens_per_second = 1.0 / b1_baseline_time_s

    return {
        "case_id": label,
        "weight_bytes_total": weight_bytes_total,
        "tree_tokens_k": k_tokens,
        "verify_flops": verify_flops,
        "arithmetic_intensity_flops_per_byte": round(arithmetic_intensity, 6),
        "ridge_flops_per_byte": round(ridge_flops_per_byte, 6),
        "memory_time_lower_bound_seconds": round(memory_time_s, 9),
        "compute_time_lower_bound_seconds": round(compute_time_s, 9),
        "total_step_time_seconds": round(verify_step_time_s, 9),
        "regime": regime,
        "effective_tokens_per_second": round(effective_tokens_per_second, 6),
        "speedup_over_b1_greedy": round(effective_tokens_per_second / baseline_tokens_per_second, 6),
    }


def evaluate_spatial_sram_case(
    *,
    case_id: str,
    architecture: str,
    total_weight_bytes: int,
    on_chip_sram_bytes_per_unit: int,
    usable_weight_fraction: float,
    stream_bandwidth_bytes_per_sec_per_unit: int,
    inter_unit_hop_latency_ns: int,
) -> dict:
    """Evaluate unit count and SRAM weight-streaming floor for Groq LPU vs Cerebras WSE."""
    label = _ascii_label(case_id, "case_id")
    arch = _ascii_label(architecture, "architecture")
    weights = integer(total_weight_bytes, "total_weight_bytes")
    sram_unit = integer(on_chip_sram_bytes_per_unit, "on_chip_sram_bytes_per_unit")
    stream_bw = integer(stream_bandwidth_bytes_per_sec_per_unit, "stream_bandwidth_bytes_per_sec_per_unit")
    hop_ns = integer(inter_unit_hop_latency_ns, "inter_unit_hop_latency_ns")
    if (
        not isinstance(usable_weight_fraction, (int, float))
        or isinstance(usable_weight_fraction, bool)
        or not (0.0 < usable_weight_fraction <= 1.0)
    ):
        raise ValueError("usable_weight_fraction: expected float in (0, 1]")

    usable_bytes_per_unit = int(sram_unit * usable_weight_fraction)
    if usable_bytes_per_unit <= 0:
        raise ValueError("usable_bytes_per_unit must be positive")
    min_units_for_pure_sram = math.ceil(weights / usable_bytes_per_unit)
    fits_single_unit_sram = min_units_for_pure_sram == 1
    weights_per_unit_if_sharded = math.ceil(weights / min_units_for_pure_sram)
    sram_stream_time_per_token_s = weights_per_unit_if_sharded / stream_bw
    daisy_chain_hop_time_s = ((min_units_for_pure_sram - 1) * hop_ns) / 1e9

    return {
        "case_id": label,
        "architecture": arch,
        "usable_sram_bytes_per_unit": usable_bytes_per_unit,
        "min_units_for_pure_sram_residency": min_units_for_pure_sram,
        "fits_single_unit_sram": fits_single_unit_sram,
        "sram_weight_stream_time_seconds": round(sram_stream_time_per_token_s, 9),
        "inter_unit_propagation_floor_seconds": round(daisy_chain_hop_time_s, 9),
        "combined_token_latency_floor_seconds": round(
            sram_stream_time_per_token_s + daisy_chain_hop_time_s, 9
        ),
    }


def calculate(fixture: dict) -> dict:
    """Evaluate all innovator architecture tradeoff cases."""
    if (
        not isinstance(fixture, dict)
        or type(fixture.get("schema_version")) is not int
        or fixture["schema_version"] != 1
        or fixture.get("synthetic") is not True
        or not isinstance(fixture.get("kvouter_cases"), list)
        or not fixture["kvouter_cases"]
        or not isinstance(fixture.get("speculative_tree_cases"), list)
        or not fixture["speculative_tree_cases"]
        or not isinstance(fixture.get("spatial_sram_cases"), list)
        or not fixture["spatial_sram_cases"]
    ):
        raise ValueError("fixture: expected version-one synthetic innovator cases")

    return {
        "schema_version": 1,
        "evidence_status": "own-offline",
        "synthetic": True,
        "kvouter_cases": [
            {"inputs": c, "result": evaluate_kvouter_case(**c)} for c in fixture["kvouter_cases"]
        ],
        "speculative_tree_cases": [
            {"inputs": c, "result": evaluate_speculative_tree_case(**c)}
            for c in fixture["speculative_tree_cases"]
        ],
        "spatial_sram_cases": [
            {"inputs": c, "result": evaluate_spatial_sram_case(**c)}
            for c in fixture["spatial_sram_cases"]
        ],
        "limitations": LIMITATIONS,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate results.json or verify exact match with --check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        content = json.dumps(calculate(fixture), indent=2, sort_keys=True, allow_nan=False) + "\n"
        output = HERE / "results.json"
        if args.check:
            if not output.is_file() or output.read_bytes() != content.encode("utf-8"):
                print("Innovator arch tradeoffs check failed: missing or different results.json", file=sys.stderr)
                return 1
        else:
            with output.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        print("Innovator arch tradeoffs failed: invalid fixture or output", file=sys.stderr)
        return 1
    print(
        "own-offline innovator arch tradeoffs verified"
        if args.check
        else "own-offline innovator arch tradeoffs regenerated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
