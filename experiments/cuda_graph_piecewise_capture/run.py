#!/usr/bin/env python3
"""Own-offline analytical model for Piecewise & Breakable CUDA Graph capture vs Eager and Full Graph."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "input.json"
RESULTS_JSON_PATH = ROOT / "results.json"
RESULTS_CSV_PATH = ROOT / "results.csv"


def _pad_to_bucket(batch_size: int, buckets: list[int]) -> int:
    idx = bisect.bisect_left(buckets, batch_size)
    return buckets[idx] if idx < len(buckets) else batch_size


def _gpu_compute_time_ms(
    batch_size: int,
    model_cfg: dict[str, Any],
    gpu_cfg: dict[str, Any],
) -> tuple[float, float]:
    """Return (non_attn_gpu_ms, attn_gpu_ms) for a given batch size."""
    weight_read_ms = (model_cfg["active_params_gb_per_gpu"] / gpu_cfg["hbm_bw_gbps"]) * 1000.0
    flops_per_token_gb = model_cfg["active_params_gb_per_gpu"] * 2.0
    gemm_compute_ms = (batch_size * flops_per_token_gb) / gpu_cfg["tflops"]
    non_attn_gpu_ms = (gpu_cfg["kernel_base_us"] / 1000.0) + max(
        weight_read_ms, gemm_compute_ms * 1.15
    ) + 0.0042 * batch_size

    kv_bytes = (
        batch_size
        * model_cfg["avg_context_tokens"]
        * model_cfg["kv_bytes_per_token_per_gpu"]
    )
    kv_gb = kv_bytes / 1e9
    attn_gpu_ms = 0.14 + (kv_gb / (gpu_cfg["hbm_bw_gbps"] * 0.78)) * 1000.0
    return non_attn_gpu_ms, attn_gpu_ms


def evaluate_cuda_graph_capture(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    buckets = sorted(config["capture_buckets"])
    mem_cfg = config["memory_pool"]

    eager_cpu_ms = (model_cfg["eager_ops_per_step"] * model_cfg["cpu_launch_us_per_op"]) / 1000.0
    piecewise_cpu_ms = (
        model_cfg["piecewise_graph_segments"] * model_cfg["graph_replay_us_per_segment"]
        + model_cfg["eager_attention_ops_per_step"] * model_cfg["cpu_launch_us_per_op"]
    ) / 1000.0
    full_graph_cpu_ms = model_cfg["full_graph_replay_us"] / 1000.0

    # Memory pool comparison across 32 capture buckets
    num_layers = model_cfg["num_layers"]
    bytes_per_tok_layer = mem_cfg["intermediate_bytes_per_token_per_layer"]
    # Without shared pool: every bucket and every layer allocates private intermediate buffers
    unshared_bytes = sum(
        b * bytes_per_tok_layer * num_layers for b in buckets
    )
    unshared_pool_gb = (unshared_bytes / (1024.0**3)) + (mem_cfg["persistent_io_buffer_mb"] / 1024.0)

    # With vLLM global_graph_pool + weak_ref_tensors(output):
    # Live peak across buckets is bounded by max(buckets) * 2 ping-pong layer slots + metadata
    max_b = max(buckets)
    shared_live_bytes = max_b * bytes_per_tok_layer * 4
    metadata_mb = (
        len(buckets)
        * model_cfg["piecewise_graph_segments"]
        * mem_cfg["graph_handle_metadata_mb_per_segment"]
    )
    shared_pool_gb = (
        (shared_live_bytes / (1024.0**3))
        + (mem_cfg["persistent_io_buffer_mb"] / 1024.0)
        + (metadata_mb / 1024.0)
    )
    memory_savings_ratio = unshared_pool_gb / shared_pool_gb

    rows: list[dict[str, Any]] = []
    for gpu_name, gpu_cfg in config["gpus"].items():
        for b_unpadded in config["probe_unpadded_batches"]:
            b_padded = _pad_to_bucket(b_unpadded, buckets)
            padding_tokens = b_padded - b_unpadded
            padding_waste_pct = (padding_tokens / b_padded) * 100.0

            non_attn_unpadded_ms, attn_unpadded_ms = _gpu_compute_time_ms(
                b_unpadded, model_cfg, gpu_cfg
            )
            non_attn_padded_ms, attn_padded_ms = _gpu_compute_time_ms(
                b_padded, model_cfg, gpu_cfg
            )

            # Eager mode: CPU launch bottleneck when CPU launch exceeds GPU execution
            # Using 35% host-device overlap for eager execution
            eager_exposed_cpu_ms = max(
                0.0, eager_cpu_ms - 0.35 * (non_attn_unpadded_ms + attn_unpadded_ms)
            )
            eager_tpot_ms = non_attn_unpadded_ms + attn_unpadded_ms + eager_exposed_cpu_ms

            # Piecewise / Breakable CUDA Graph:
            # Non-attention subgraphs use padded bucket b_padded; eager attention runs at exact b_unpadded!
            piecewise_exposed_cpu_ms = max(
                0.0, piecewise_cpu_ms - 0.55 * (non_attn_padded_ms + attn_unpadded_ms)
            )
            piecewise_tpot_ms = (
                non_attn_padded_ms + attn_unpadded_ms + piecewise_exposed_cpu_ms
            )

            # Full CUDA Graph: both non-attention and attention must pad to b_padded
            full_tpot_ms = non_attn_padded_ms + attn_padded_ms + full_graph_cpu_ms

            padding_compute_penalty_ms = non_attn_padded_ms - non_attn_unpadded_ms
            cpu_launch_saved_ms = eager_exposed_cpu_ms - piecewise_exposed_cpu_ms

            rows.append(
                {
                    "gpu": gpu_name,
                    "unpadded_bs": b_unpadded,
                    "padded_bs": b_padded,
                    "padding_tokens": padding_tokens,
                    "padding_waste_pct": round(padding_waste_pct, 2),
                    "eager_cpu_launch_ms": round(eager_cpu_ms, 3),
                    "eager_tpot_ms": round(eager_tpot_ms, 3),
                    "piecewise_tpot_ms": round(piecewise_tpot_ms, 3),
                    "full_graph_tpot_ms": round(full_tpot_ms, 3),
                    "piecewise_speedup_vs_eager": round(eager_tpot_ms / piecewise_tpot_ms, 2),
                    "full_speedup_vs_eager": round(eager_tpot_ms / full_tpot_ms, 2),
                    "padding_compute_penalty_ms": round(padding_compute_penalty_ms, 3),
                    "cpu_launch_saved_ms": round(cpu_launch_saved_ms, 3),
                }
            )

    b200_bs1 = next(
        r for r in rows if r["gpu"] == "B200_NVFP4" and r["unpadded_bs"] == 1
    )
    h100_bs33 = next(
        r for r in rows if r["gpu"] == "H100_FP8" and r["unpadded_bs"] == 33
    )
    b200_bs257 = next(
        r for r in rows if r["gpu"] == "B200_NVFP4" and r["unpadded_bs"] == 257
    )

    summary = {
        "num_capture_buckets": len(buckets),
        "eager_cpu_launch_ms": round(eager_cpu_ms, 3),
        "piecewise_cpu_launch_ms": round(piecewise_cpu_ms, 3),
        "full_graph_cpu_launch_ms": round(full_graph_cpu_ms, 3),
        "unshared_graph_pool_gb": round(unshared_pool_gb, 2),
        "shared_global_graph_pool_gb": round(shared_pool_gb, 2),
        "memory_pool_compression_ratio": round(memory_savings_ratio, 2),
        "b200_bs1_eager_tpot_ms": b200_bs1["eager_tpot_ms"],
        "b200_bs1_piecewise_tpot_ms": b200_bs1["piecewise_tpot_ms"],
        "b200_bs1_full_tpot_ms": b200_bs1["full_graph_tpot_ms"],
        "b200_bs1_piecewise_speedup": b200_bs1["piecewise_speedup_vs_eager"],
        "h100_bs33_padded_to_40_piecewise_tpot_ms": h100_bs33["piecewise_tpot_ms"],
        "h100_bs33_full_graph_tpot_ms": h100_bs33["full_graph_tpot_ms"],
        "b200_bs257_piecewise_vs_full_delta_ms": round(
            b200_bs257["full_graph_tpot_ms"] - b200_bs257["piecewise_tpot_ms"], 3
        ),
    }

    return {
        "experiment_id": "cuda_graph_piecewise_capture",
        "evidence_class": "own-offline",
        "summary": summary,
        "rows": rows,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    results = evaluate_cuda_graph_capture(config)
    rendered = json.dumps(results, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not RESULTS_JSON_PATH.exists():
            raise SystemExit("Missing results.json")
        existing = RESULTS_JSON_PATH.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        print("cuda_graph_piecewise_capture check passed.")
        return 0

    RESULTS_JSON_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    _write_csv(results["rows"], RESULTS_CSV_PATH)
    print("cuda_graph_piecewise_capture results written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
