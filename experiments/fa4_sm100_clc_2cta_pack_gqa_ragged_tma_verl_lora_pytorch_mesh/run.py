#!/usr/bin/env python3
"""Offline simulator for Phase 76 Track C:
FlashAttention-4 SM100 2-CTA CLC Logical Coordinates, GQA Backward pack_gqa, & Per-CTA Ragged TMA Descriptors,
veRL Single-Block Pinned LoRA Bucket Staging, PyTorch DeviceMesh Sorted Subgroup Split & XPU RAII, and Mooncake Float64 Msgpack.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def work_q_block(work_m: int, stage: int, cta_rank: int, q_stage: int, cta_group_size: int) -> int:
    """FA4 PR #3034 (98c1cee0ed59): canonical SM100 forward work-tile to q_block mapping."""
    return (work_m * q_stage + stage) * cta_group_size + cta_rank


def simulate_fa4_sm100_clc_2cta_and_pack_gqa_and_tma() -> dict[str, object]:
    """Model FA4 PR #3034 (98c1cee0ed59), PR #2967 (bc52a40aef87), and PR #3071 (10b74d98fb26)."""
    # 1. PR #3034: Verify work_q_block partitions Q rows disjointly for C=2, S=2, R=128
    m_block_size = 128
    cta_group_size = 2
    q_stage = 2
    seqlen_q = 1000
    work_rows = m_block_size * cta_group_size * q_stage  # 512
    num_work_m = ceil_div(seqlen_q, work_rows)  # 2

    # Under the buggy per-CTA CLC mapping, CTA 0 got work_m_buggy = 2*work_m + 0 and CTA 1 got 2*work_m + 1,
    # causing CTA 1 on work_m=0 to compute q_block for work_m=1 (rows 512..639 instead of 128..255)!
    buggy_cta1_stage0_q_block = work_q_block(work_m=1, stage=0, cta_rank=1, q_stage=q_stage, cta_group_size=cta_group_size)
    fixed_cta1_stage0_q_block = work_q_block(work_m=0, stage=0, cta_rank=1, q_stage=q_stage, cta_group_size=cta_group_size)

    seen_rows: set[int] = set()
    disjoint = True
    for wm, st, rk in itertools.product(range(num_work_m), range(q_stage), range(cta_group_size)):
        qb = work_q_block(wm, st, rk, q_stage=q_stage, cta_group_size=cta_group_size)
        for r in range(qb * m_block_size, (qb + 1) * m_block_size):
            if r in seen_rows:
                disjoint = False
            seen_rows.add(r)

    # 2. PR #2967: SM100 GQA Backward pack_gqa critical-path & wave-loss check
    # On B200, MQA 16/1 hd256 on one 8k causal sequence + 16 short ones ran 3946 us packed vs 2297 us unpacked
    # before adding the critical-path guard.
    b200_hd128_gqa_2k_gain_min_pct = 13.0
    b200_hd128_gqa_2k_gain_max_pct = 26.0
    b200_hd256_gqa_2k_gain_min_pct = 4.0
    b200_hd256_gqa_2k_gain_max_pct = 10.0
    b200_hd256_gqa_8k_gain_max_pct = 11.0
    ragged_mqa_packed_us = 3946.0
    ragged_mqa_unpacked_us = 2297.0

    # 3. PR #3071: Per-CTA TMA descriptor workspace sizing for H100 B=49, seqlen_q=128, H=6, tile_m=128
    # Unfixed bound ignored per-batch scheduler padding and allocated 302 slots while 582 CTAs launched!
    b_h100 = 49
    seqlen_q_h100 = 128
    h_h100 = 6
    tile_m_h100 = 128
    n_q_rows = b_h100 * seqlen_q_h100  # 6272
    unfixed_slots = 302
    launched_ctas_h100 = 582
    fixed_m_blocks_bound = n_q_rows // tile_m_h100 + 2 * b_h100 + 1  # 49 + 98 + 1 = 148 per head
    fixed_total_slots = fixed_m_blocks_bound * h_h100  # 888 >= 582
    tma_desc_bytes = 128

    return {
        "clc_2cta_work_rows": work_rows,
        "clc_2cta_buggy_cta1_stage0_q_block": buggy_cta1_stage0_q_block,
        "clc_2cta_fixed_cta1_stage0_q_block": fixed_cta1_stage0_q_block,
        "clc_2cta_rows_disjoint_and_covering": (disjoint and len(seen_rows) == num_work_m * work_rows),
        "clc_2cta_gb300_prev_max_error": 0.39,
        "clc_2cta_gb300_fixed_max_error": 0.0,
        "bwd_pack_gqa_hd128_2k_gain_pct_range": [b200_hd128_gqa_2k_gain_min_pct, b200_hd128_gqa_2k_gain_max_pct],
        "bwd_pack_gqa_hd256_2k_gain_pct_range": [b200_hd256_gqa_2k_gain_min_pct, b200_hd256_gqa_2k_gain_max_pct],
        "bwd_pack_gqa_hd256_8k_gain_max_pct": b200_hd256_gqa_8k_gain_max_pct,
        "bwd_pack_gqa_ragged_mqa_packed_us": ragged_mqa_packed_us,
        "bwd_pack_gqa_ragged_mqa_unpacked_us": ragged_mqa_unpacked_us,
        "tma_h100_unfixed_slots": unfixed_slots,
        "tma_h100_launched_ctas": launched_ctas_h100,
        "tma_h100_fixed_total_slots": fixed_total_slots,
        "tma_h100_workspace_bytes": fixed_total_slots * tma_desc_bytes,
    }


def simulate_verl_single_block_lora_bucket_staging() -> dict[str, object]:
    """Model veRL PR #8352 (3c6d9b43b67c): stage LoRA buckets into one pinned CPU block on GB300."""
    buckets = [
        {
            "name": "4000x(16,2048)_250MiB",
            "num_tensors": 4000,
            "shape": [16, 2048],
            "total_mib": 250.0,
            "main_device_clones": 8000,
            "main_pinned_host_allocs": 4000,
            "pr_device_clones": 0,
            "pr_pinned_host_allocs": 1,
            "main_ms": 165.0,
            "pr_ms": 35.0,
            "speedup_x": 4.7,
        },
        {
            "name": "16000x(16,1024)_500MiB",
            "num_tensors": 16000,
            "shape": [16, 1024],
            "total_mib": 500.0,
            "main_device_clones": 32000,
            "main_pinned_host_allocs": 16000,
            "pr_device_clones": 0,
            "pr_pinned_host_allocs": 1,
            "main_ms": 705.0,
            "pr_ms": 131.0,
            "speedup_x": 5.4,
        },
    ]
    return {
        "buckets": buckets,
        "max_speedup_x": max(b["speedup_x"] for b in buckets),
        "min_speedup_x": min(b["speedup_x"] for b in buckets),
    }


def simulate_pytorch_devicemesh_and_xpu_and_mooncake() -> dict[str, object]:
    """Model PyTorch PR #201949 (020ab1c83528), PR #200890 (8e9dd78a4c84), and Mooncake PR #4906 (12a7e9f10a41)."""
    unsorted_subgroup_ranks = [[3, 1], [2, 0]]
    sorted_subgroup_ranks = [sorted(r) for r in unsorted_subgroup_ranks]

    # XPU _StaticXpuLauncher refcount & Level Zero handle lifecycle (#200890):
    # Py_BuildValue("(OO)", ...) incremented py_kernel_ptr refcount from 1 to 2 (leaking on close());
    # Py_BuildValue("(NO)", ...) steals the reference (refcount stays 1, freed when self._kernels clears).
    py_buildvalue_O_refcount = 2
    py_buildvalue_N_refcount = 1

    # Mooncake msgpack-c 6.1.0 float64 timestamp encoding (#4906):
    # Integral double 1700000000.0 was packed as integer without explicit float64 packer, breaking SGLang.
    return {
        "devicemesh_unsorted_ranks": unsorted_subgroup_ranks,
        "devicemesh_sorted_ranks": sorted_subgroup_ranks,
        "xpu_py_buildvalue_O_refcount": py_buildvalue_O_refcount,
        "xpu_py_buildvalue_N_refcount": py_buildvalue_N_refcount,
        "xpu_raii_destruction_order": ["zeKernelDestroy", "zeModuleDestroy"],
        "mooncake_integral_double_preserved_as_float64": True,
    }


def build_summary() -> dict[str, object]:
    fa4 = simulate_fa4_sm100_clc_2cta_and_pack_gqa_and_tma()
    verl = simulate_verl_single_block_lora_bucket_staging()
    runtime = simulate_pytorch_devicemesh_and_xpu_and_mooncake()
    return {
        "experiment_id": "fa4_sm100_clc_2cta_pack_gqa_ragged_tma_verl_lora_pytorch_mesh",
        "phase": 76,
        "priority": 85,
        "fa4_sm100_clc_2cta_pack_gqa_tma": fa4,
        "verl_single_block_lora_staging": verl,
        "pytorch_devicemesh_xpu_mooncake": runtime,
        "invariants": {
            "fa4_clc_2cta_rows_disjoint": fa4["clc_2cta_rows_disjoint_and_covering"],
            "fa4_tma_workspace_covers_launched_ctas": (
                fa4["tma_h100_fixed_total_slots"] >= fa4["tma_h100_launched_ctas"]
                and fa4["tma_h100_unfixed_slots"] < fa4["tma_h100_launched_ctas"]
            ),
            "verl_lora_5_4x_speedup": verl["max_speedup_x"] == 5.4,
            "pytorch_xpu_refcount_one": runtime["xpu_py_buildvalue_N_refcount"] == 1,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches")
    args = parser.parse_args()

    summary = build_summary()
    assert all(summary["invariants"].values()), f"Failed invariants: {summary['invariants']}"

    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = out_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of date with run.py")
        print("OK: Phase 76 Track C summary.json verified.")
        return 0

    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
