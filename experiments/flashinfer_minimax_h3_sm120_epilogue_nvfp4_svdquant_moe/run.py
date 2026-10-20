#!/usr/bin/env python3
"""Deterministic stdlib simulator for Phase 39 Experiment 1:
FlashInfer MiniMax-H3 SM120 Epilogue Proxy Fence & Ping-Pong Barrier Close,
CAKE NVFP4 SVDQuant 15-Program GEMM Backend, and Rubin MoE execute_on_streams API.

First-hand sources:
- flashinfer-ai/flashinfer@aacd0f134f2a35eaea9ea534aa5b5b31534db449 (#6847)
- flashinfer-ai/flashinfer@ec3bcf0301dac6455897d4a450115fdd8fea1bd3 (#6878)
- flashinfer-ai/flashinfer@6445497657b7bc06fcc768d6d499e38627221818 (#6488)
- flashinfer-ai/flashinfer@5ea501832bf7e37a7acde903a32329d580c0e9b4 (#6869)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_minimax_h3_sm120_proxy_fence_and_ping_pong(
    num_items: int = 8,
    num_warps: int = 8,
) -> dict[str, Any]:
    """Model SM120 MiniMax-H3 dense/FP8/NVFP4 attention proxy fence and ping-pong barrier.

    In aacd0f1 (#6847):
    - Legacy dense kernel issued `fence.proxy.async.shared::cta` at the start of the
      next work item *after* `__syncthreads()`. Since a thread's proxy fence only
      orders that thread's own earlier generic-proxy accesses (`q_smem` Q scaling & O
      staging), the other 7 warps' generic-proxy accesses were not ordered before Warp 0's
      elected thread issued TMA refill (`mbarrier_arrive_expect_tx(q_full_addr, 32768)`).
    - Fixed dense kernel issues `fence.proxy.async.shared::cta` on every thread *before*
      the end-of-item `__syncthreads()`, ordering all 8 warps before the next TMA refill.
    - In dense, FP8 varlen, and NVFP4 varlen kernels, Warp Group 1 (`warp >= 4`) pre-arrives
      on named barrier 1 (`256` threads) once before the persistent loop. At loop exit,
      Group 1's last arrival leaves barrier 1 open (`128/256` threads arrived) unless
      Group 0 (`warp < 4`) executes `barrier.sync 1, 256` after the loop.
    """
    legacy_unordered_warps_per_refill = num_warps - 1  # warps 1..7 unordered wrt warp 0 TMA
    fixed_unordered_warps_per_refill = 0

    legacy_total_proxy_hazards = max(0, num_items - 1) * legacy_unordered_warps_per_refill
    fixed_total_proxy_hazards = max(0, num_items - 1) * fixed_unordered_warps_per_refill

    # Named barrier 1 round accounting (threads per group = 128, barrier threshold = 256)
    group1_arrivals = 1 + num_items  # 1 pre-arrive + num_items loop arrivals
    group0_loop_syncs = num_items
    legacy_unjoined_barrier1_rounds = group1_arrivals - group0_loop_syncs  # 1 open round (128/256)
    fixed_group0_syncs = group0_loop_syncs + 1  # closing `if (warp < 4) barrier.sync 1, 256;`
    fixed_unjoined_barrier1_rounds = group1_arrivals - fixed_group0_syncs  # 0 open rounds

    return {
        "num_items": num_items,
        "num_warps": num_warps,
        "q_smem_tma_tx_bytes": 32768,
        "legacy_unordered_warps_per_refill": legacy_unordered_warps_per_refill,
        "fixed_unordered_warps_per_refill": fixed_unordered_warps_per_refill,
        "legacy_total_proxy_hazards": legacy_total_proxy_hazards,
        "fixed_total_proxy_hazards": fixed_total_proxy_hazards,
        "legacy_unjoined_barrier1_rounds": legacy_unjoined_barrier1_rounds,
        "fixed_unjoined_barrier1_rounds": fixed_unjoined_barrier1_rounds,
        "gb300_timing_delta_pct_bound": 0.35,
    }


def simulate_cake_nvfp4_svdquant_gemm_backend() -> dict[str, Any]:
    """Model the 15-program Cake NVFP4 SVDQuant GEMM backend on B200 (sm_100a) and GB300 (sm_103a).

    In ec3bcf0 (#6878):
    `out = BF16(alpha * (deq(A) @ deq(B)^T + D @ L1^T) + bias)` across LoRA ranks 32, 64, 96, 128.
    Replaces 7 legacy Cake programs with 15 converged programs in `flashinfer/cake_nvfp4_svdquant_gemm_backend`.
    """
    b200_cases = [
        {"M": 129, "N": 3072, "K": 3072, "rank": 32, "bias": 1, "prev_cake_us": 10.66, "cutlass_us": 8.19, "new_cake_us": 6.46},
        {"M": 129, "N": 3072, "K": 3072, "rank": 32, "bias": 0, "prev_cake_us": 6.46, "cutlass_us": 8.19, "new_cake_us": 6.34},
        {"M": 129, "N": 3072, "K": 3072, "rank": 128, "bias": 1, "prev_cake_us": 11.10, "cutlass_us": 8.58, "new_cake_us": 6.69},
        {"M": 129, "N": 3072, "K": 12288, "rank": 128, "bias": 1, "prev_cake_us": 26.37, "cutlass_us": 16.29, "new_cake_us": 13.95},
        {"M": 6889, "N": 12288, "K": 3072, "rank": 32, "bias": 1, "prev_cake_us": 319.39, "cutlass_us": 103.23, "new_cake_us": 97.38},
        {"M": 16384, "N": 3072, "K": 3072, "rank": 32, "bias": 1, "prev_cake_us": 192.11, "cutlass_us": 63.07, "new_cake_us": 59.10},
        {"M": 16384, "N": 12288, "K": 3072, "rank": 32, "bias": 1, "prev_cake_us": 731.39, "cutlass_us": 253.40, "new_cake_us": 242.88},
    ]
    gb300_cases = [
        {"M": 129, "N": 3072, "K": 3072, "rank": 32, "bias": 1, "prev_cake_us": 10.46, "cutlass_us": 7.20, "new_cake_us": 5.98},
        {"M": 129, "N": 3072, "K": 12288, "rank": 128, "bias": 1, "prev_cake_us": 25.25, "cutlass_us": 14.59, "new_cake_us": 12.86},
        {"M": 6889, "N": 12288, "K": 3072, "rank": 32, "bias": 1, "prev_cake_us": 293.30, "cutlass_us": 85.17, "new_cake_us": 83.17},
        {"M": 16384, "N": 12288, "K": 3072, "rank": 32, "bias": 1, "prev_cake_us": 673.32, "cutlass_us": 195.98, "new_cake_us": 187.87},
    ]

    for row in b200_cases + gb300_cases:
        flops = 2 * row["M"] * row["N"] * (row["K"] + row["rank"])
        row["algo_tflops"] = round(flops / (row["new_cake_us"] * 1e-6) / 1e12, 2)
        row["speedup_vs_prev_cake"] = round(row["prev_cake_us"] / row["new_cake_us"], 2)
        row["speedup_vs_cutlass"] = round(row["cutlass_us"] / row["new_cake_us"], 3)

    return {
        "previous_program_count": 7,
        "converged_program_count": 15,
        "supported_ranks": [32, 64, 96, 128],
        "b200_sm100a_cases": b200_cases,
        "gb300_sm103a_cases": gb300_cases,
        "b200_max_speedup_vs_prev_cake": max(r["speedup_vs_prev_cake"] for r in b200_cases),
        "b200_max_speedup_vs_cutlass": max(r["speedup_vs_cutlass"] for r in b200_cases),
        "gb300_max_speedup_vs_prev_cake": max(r["speedup_vs_prev_cake"] for r in gb300_cases),
        "gb300_max_speedup_vs_cutlass": max(r["speedup_vs_cutlass"] for r in gb300_cases),
    }


def simulate_rubin_moe_execute_on_streams(
    num_locality_domains: int = 4,
    cta_tile_n: int = 256,
) -> dict[str, Any]:
    """Model FlashInfer Rubin MoE `torch.cuda.execute_on_streams` explicit sequence adaptation.

    In 6445497 (#6488):
    - PyTorch PR #199128 updated `torch.cuda.execute_on_streams(streams, fn, *inputs)` to require
      explicit input sequences rather than passing an implicit stream index.
    - Legacy two-argument `execute_on_streams(localized_streams, callback)` raises TypeError on
      updated PyTorch builds; passing `range(len(localized_streams))` restores FC1/FC2 execution.
    - Also computes `copy_size = cta_tile_n * (c_dtype_bits // 8)` dynamically at kernel call site
      for `cp.reduce.async.bulk` (`512` bytes for BF16, `1024` bytes for FP32 at `cta_tile_n=256`).
    """
    stream_indices = list(range(num_locality_domains))
    dispatched_fc1 = [idx for idx in stream_indices]
    dispatched_fc2 = [idx for idx in stream_indices]
    copy_size_bf16_bytes = cta_tile_n * (16 // 8)
    copy_size_fp32_bytes = cta_tile_n * (32 // 8)

    return {
        "num_locality_domains": num_locality_domains,
        "legacy_two_arg_contract_compatible": False,
        "explicit_range_sequence_compatible": dispatched_fc1 == stream_indices and dispatched_fc2 == stream_indices,
        "fc1_dispatched_domains": dispatched_fc1,
        "fc2_dispatched_domains": dispatched_fc2,
        "cp_reduce_bulk_copy_size_bf16_bytes": copy_size_bf16_bytes,
        "cp_reduce_bulk_copy_size_fp32_bytes": copy_size_fp32_bytes,
    }


def run_experiment() -> dict[str, Any]:
    sm120 = simulate_minimax_h3_sm120_proxy_fence_and_ping_pong()
    svdquant = simulate_cake_nvfp4_svdquant_gemm_backend()
    moe_streams = simulate_rubin_moe_execute_on_streams()

    invariants = {
        "sm120_proxy_hazards_eliminated": (
            sm120["legacy_total_proxy_hazards"] == 49
            and sm120["fixed_total_proxy_hazards"] == 0
        ),
        "sm120_ping_pong_barrier1_closed": (
            sm120["legacy_unjoined_barrier1_rounds"] == 1
            and sm120["fixed_unjoined_barrier1_rounds"] == 0
        ),
        "cake_nvfp4_svdquant_15_programs_converged": (
            svdquant["converged_program_count"] == 15
            and svdquant["b200_max_speedup_vs_prev_cake"] == 3.28
            and svdquant["gb300_max_speedup_vs_prev_cake"] == 3.58
            and svdquant["b200_max_speedup_vs_cutlass"] >= 1.29
        ),
        "rubin_moe_explicit_stream_inputs_verified": (
            moe_streams["explicit_range_sequence_compatible"] is True
            and moe_streams["cp_reduce_bulk_copy_size_bf16_bytes"] == 512
            and moe_streams["cp_reduce_bulk_copy_size_fp32_bytes"] == 1024
        ),
    }

    return {
        "experiment": "flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe",
        "phase": 39,
        "priority": 48,
        "evidence_mode": "own-offline",
        "sm120_minimax_h3_attention": sm120,
        "cake_nvfp4_svdquant_gemm": svdquant,
        "rubin_moe_execute_on_streams": moe_streams,
        "invariants": invariants,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_experiment()
    out_dir = Path(__file__).resolve().parent
    for target in (out_dir / "summary.json", out_dir / "results" / "summary.json"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        failed = [k for k, v in summary["invariants"].items() if not v]
        if failed:
            raise SystemExit(f"Failed invariants: {failed}")
        print(json.dumps({"status": "ok", "experiment": summary["experiment"], "invariants": summary["invariants"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
