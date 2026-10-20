#!/usr/bin/env python3
"""Deterministic simulation of DeepGEMM SM100 MegaMoE & MegaMHC (Multi-Head Hyper-Connections) JIT Kernels.

Models `deepseek-ai/DeepGEMM` (`commit 5081f8513af57e61696de4f4f04b0e707ffb503e`):
- `csrc/jit_kernels/heuristics/mega_moe.hpp`:
  - `get_block_config_for_mega_moe`: 1-sigma routing margin `num_covered_tokens = E + sqrt(E)`,
    `block_m in {16, 32, 64, 128, 192, 240}`, and asymmetric `store_block_m_l1` / `store_block_m_l2`
    (`24` and `8` at `block_m=240, MXFP8FP4` vs `40` and `40` at `BF16`).
  - `get_pipeline_config_for_mega_moe`: Exact SM100 `232,448 B` shared-memory budget (`smem_dispatch_size`,
    `smem_cd`, `smem_amax_reduction`, `smem_barriers`, `smem_task_info`, `smem_size_per_stage`) and TMA
    stage count `num_stages = (smem_capacity - smem_fixed) // smem_size_per_stage`.
- `csrc/apis/mega_mhc.hpp` & `csrc/jit_kernels/impls/sm100_mega_mhc.hpp`:
  - `get_num_splits`: dynamic Split-K `ceil_div(num_k_blocks, ceil_div(num_k_blocks, max_num_splits))`
  - Fused Multi-Head Hyper-Connections (`residual [T, hc_mult, H]`) + Sinkhorn-Knopp doubly-stochastic
    mixing + RMSNorm + Simultaneous triple FP8 `UE8M0` (`gran_k=32`) scaling-factor layouts
    (`y_gemm_sf`, `y_routed_sf`, `y_shared_sf`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SM100_SMEM_CAPACITY = 232448  # 227 KiB per SM on Blackwell SM100
SMEM_ALIGNMENT = 1024


def align_up(val: int, alignment: int) -> int:
    return ((val + alignment - 1) // alignment) * alignment


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_sf_uttcp_aligned_block_sizes(block_m: int, block_n: int) -> tuple[int, int]:
    """UTCCP 128-element aligned SF block sizes on SM100."""
    # load_block_m = block_m // 2 for 2-CTA multicast on A
    return align_up(block_m // 2, 128), align_up(block_n, 128)


def get_block_config_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_tokens: int,
    mma_kind: str,
) -> dict[str, int]:
    """Exact reproduction of `get_block_config_for_mega_moe` in `csrc/jit_kernels/heuristics/mega_moe.hpp`."""
    num_expected_tokens = float(num_tokens) * num_ranks * num_topk / num_experts
    num_covered_tokens = num_expected_tokens + math.sqrt(num_expected_tokens)

    block_m = 16 if num_expected_tokens <= 10 else 32
    if num_expected_tokens > 24:
        num_blocks = int(math.ceil(num_covered_tokens / 240.0))
        if num_blocks == 1 and num_covered_tokens > 192 and (num_experts // num_ranks) < 14:
            num_blocks = 2
        for candidate in (64, 128, 192, 240):
            block_m = candidate
            if num_blocks * candidate >= num_covered_tokens:
                break

    if block_m <= 16:
        store_block_m_l1 = 8
    elif block_m <= 64:
        store_block_m_l1 = 16
    elif block_m <= 192:
        store_block_m_l1 = 32
    else:
        store_block_m_l1 = 24 if mma_kind == "MXFP8FP4" else 40

    store_block_m_l2 = 8 if (block_m == 240 and mma_kind == "MXFP8FP4") else store_block_m_l1
    elem_bytes = 1 if mma_kind == "MXFP8FP4" else 2
    block_k = 128 // elem_bytes

    return {
        "cluster_size": 2,
        "block_m": block_m,
        "store_block_m_l1": store_block_m_l1,
        "store_block_m_l2": store_block_m_l2,
        "block_k": block_k,
        "num_epilogue_threads": 256,
        "expected_tokens_per_expert": round(num_expected_tokens, 2),
        "covered_tokens_1sigma": round(num_covered_tokens, 2),
    }


def get_pipeline_config_for_mega_moe(
    num_experts: int,
    hidden: int,
    block_m: int,
    store_block_m_l1: int,
    store_block_m_l2: int,
    mma_kind: str,
) -> dict[str, int]:
    """Exact reproduction of `get_pipeline_config_for_mega_moe` in `csrc/jit_kernels/heuristics/mega_moe.hpp`."""
    is_sf = mma_kind == "MXFP8FP4"
    elem_bytes = 1 if is_sf else 2
    block_n = 128
    block_k = 128 // elem_bytes
    load_block_m = block_m // 2
    gran_k = 32
    num_dispatch_warps = 4  # 128 / 32
    num_epilogue_warps = 8  # 256 / 32
    num_epilogue_warpgroups = num_epilogue_warps // 4  # 2

    num_max_pull_bytes = 8192 if is_sf else 4096
    num_bytes_per_pull = hidden * elem_bytes
    while num_bytes_per_pull > num_max_pull_bytes:
        num_bytes_per_pull //= 2

    smem_expert_count_size = align_up(num_experts * 4, SMEM_ALIGNMENT)
    # Buffer(Data(num_bytes_per_pull), num_dispatch_warps, 1) -> num_bytes_per_pull * num_dispatch_warps
    smem_send_buffers_size = align_up(num_bytes_per_pull * num_dispatch_warps, SMEM_ALIGNMENT)
    smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size

    smem_cd_l1 = num_epilogue_warpgroups * store_block_m_l1 * (block_n // 2) * 2 * elem_bytes
    smem_cd_l2 = num_epilogue_warpgroups * store_block_m_l2 * block_n * 2  # sizeof(nv_bfloat16) == 2
    smem_cd = align_up(max(smem_cd_l1, smem_cd_l2), SMEM_ALIGNMENT)

    smem_task_info = 2 * 32  # 2 schedule stages * sizeof(TaskInfo<true>)
    smem_barriers = (num_dispatch_warps + 2 * 2 + num_epilogue_warps * 2 + 2 * 2) * 8
    smem_amax_reduction = (store_block_m_l1 * num_epilogue_warps * 4) if is_sf else 0
    smem_tmem_ptr = 4

    sf_block_m, sf_block_n = get_sf_uttcp_aligned_block_sizes(block_m, block_n) if is_sf else (0, 0)
    smem_sfa_per_stage = sf_block_m * (block_k // gran_k) if is_sf else 0
    smem_sfb_per_stage = sf_block_n * (block_k // gran_k) if is_sf else 0
    smem_a_per_stage = load_block_m * block_k * elem_bytes
    smem_b_per_stage = block_n * block_k * elem_bytes
    smem_size_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage + 16

    smem_fixed = (
        smem_dispatch_size
        + smem_cd
        + smem_amax_reduction
        + smem_barriers
        + smem_task_info
        + smem_tmem_ptr
    )
    num_stages = (SM100_SMEM_CAPACITY - smem_fixed) // smem_size_per_stage
    return {
        "smem_dispatch_bytes": smem_dispatch_size,
        "smem_cd_l1_bytes": smem_cd_l1,
        "smem_cd_l2_bytes": smem_cd_l2,
        "smem_cd_aligned_bytes": smem_cd,
        "smem_fixed_bytes": smem_fixed,
        "smem_per_stage_bytes": smem_size_per_stage,
        "num_stages": num_stages,
        "total_smem_bytes": smem_fixed + num_stages * smem_size_per_stage,
    }


def get_mhc_num_splits(num_m_blocks: int, hidden: int, num_sms: int = 148) -> int:
    """Exact reproduction of `mega_mhc::get_num_splits` in `csrc/apis/mega_mhc.hpp`."""
    block_k = 128
    default_splits = 16
    max_splits_cap = 32
    num_k_blocks = hidden // block_k
    max_num_splits = min(max_splits_cap, max(default_splits, num_sms // max(1, num_m_blocks)))
    return ceil_div(num_k_blocks, ceil_div(num_k_blocks, max_num_splits))


def simulate_deepgemm_mega_moe_mhc() -> dict[str, Any]:
    num_ranks = 8
    num_experts = 128  # 16 experts per rank (>= 14 threshold for single 240-row block per expert)
    num_topk = 8
    hidden = 7168

    # Sweep token batch sizes per rank to trace block_m and TMA pipeline stages
    token_sweeps = []
    for num_tokens in (16, 80, 280, 400):
        cfg = get_block_config_for_mega_moe(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_max_tokens_per_rank=512,
            num_topk=num_topk,
            num_tokens=num_tokens,
            mma_kind="MXFP8FP4",
        )
        pipe = get_pipeline_config_for_mega_moe(
            num_experts=num_experts,
            hidden=hidden,
            block_m=cfg["block_m"],
            store_block_m_l1=cfg["store_block_m_l1"],
            store_block_m_l2=cfg["store_block_m_l2"],
            mma_kind="MXFP8FP4",
        )
        token_sweeps.append(
            {
                "num_tokens_per_rank": num_tokens,
                "expected_tokens_per_expert": cfg["expected_tokens_per_expert"],
                "covered_tokens_1sigma": cfg["covered_tokens_1sigma"],
                "block_m": cfg["block_m"],
                "store_block_m_l1": cfg["store_block_m_l1"],
                "store_block_m_l2": cfg["store_block_m_l2"],
                "smem_cd_aligned_bytes": pipe["smem_cd_aligned_bytes"],
                "smem_per_stage_bytes": pipe["smem_per_stage_bytes"],
                "num_tma_stages": pipe["num_stages"],
                "total_smem_bytes": pipe["total_smem_bytes"],
            }
        )

    # Compare Asymmetric (`store_block_m_l1=24, store_block_m_l2=8`) vs Symmetric (`store_block_m_l1=40, store_block_m_l2=40`) at block_m=240
    asym_240 = get_pipeline_config_for_mega_moe(num_experts, hidden, 240, 24, 8, "MXFP8FP4")
    sym_240 = get_pipeline_config_for_mega_moe(num_experts, hidden, 240, 40, 40, "MXFP8FP4")

    # MegaMHC (`csrc/apis/mega_mhc.hpp`) HBM traffic & latency analysis for T=4096, H=7168, hc_mult=4
    tokens = 4096
    hc_mult = 4
    bf16_b = 2
    fp8_b = 1
    # Unfused 6-kernel pipeline:
    # 1. Pre-mix & HC linear projection (read residual [T,4,H] + x [T,H], write mixes)
    # 2. Sinkhorn-Knopp 4 iterations + residual update (read/write residual [T,4,H] + y_bf16 [T,H])
    # 3. RMSNorm (read y_bf16 [T,H], write norm_out [T,H])
    # 4. Quantize FP8 + TMA-aligned gemm_sf
    # 5. Transpose/re-layout routed_sf
    # 6. Transpose/re-layout shared_sf for MegaMoE shared experts
    residual_bytes = tokens * hc_mult * hidden * bf16_b  # 234.88 MB
    act_bf16_bytes = tokens * hidden * bf16_b  # 58.72 MB
    act_fp8_bytes = tokens * hidden * fp8_b  # 29.36 MB
    sf_bytes = tokens * (hidden // 128) * 4  # 0.92 MB per SF tensor (4 UE8M0 packed per int32)

    unfused_hbm_bytes = (
        (residual_bytes + act_bf16_bytes)  # Kernel 1 read
        + (2 * residual_bytes + act_bf16_bytes)  # Kernel 2 read+write residual + write y_bf16
        + (2 * act_bf16_bytes)  # Kernel 3 RMSNorm read+write
        + (act_bf16_bytes + act_fp8_bytes + sf_bytes)  # Kernel 4 FP8 quant + gemm_sf
        + (2 * sf_bytes)  # Kernel 5 routed_sf layout
        + (2 * sf_bytes)  # Kernel 6 shared_sf layout
    )
    # Fused `sm100_mega_mhc`: single pass reads x + residual once, writes new_residual + y_bf16 + y_fp8 + routed_sf + shared_sf
    fused_hbm_bytes = (
        residual_bytes  # read residual
        + act_bf16_bytes  # read x
        + residual_bytes  # write new_residual
        + act_bf16_bytes  # write y_bf16
        + act_fp8_bytes  # write y_fp8
        + 2 * sf_bytes  # write y_routed_sf + y_shared_sf simultaneously in epilogue
    )

    b200_eff_hbm_bw_gbps = 6800.0  # GB/s effective on B200
    launch_overhead_us = 5.5
    unfused_mhc_us = 6 * launch_overhead_us + (unfused_hbm_bytes / (b200_eff_hbm_bw_gbps * 1e9)) * 1e6
    fused_mhc_us = 1 * launch_overhead_us + (fused_hbm_bytes / (b200_eff_hbm_bw_gbps * 1e9)) * 1e6

    num_m_blocks = ceil_div(tokens, 64)
    dynamic_splits = get_mhc_num_splits(num_m_blocks, hidden, num_sms=148)

    return {
        "experiment": "deepgemm_mega_moe_mhc_sim",
        "source_commit": "5081f8513af57e61696de4f4f04b0e707ffb503e",
        "mega_moe_tile_sweeps": token_sweeps,
        "asymmetric_store_block_240_comparison": {
            "asymmetric_mxfp8fp4_24_8": asym_240,
            "symmetric_fallback_40_40": sym_240,
            "smem_cd_saved_bytes": sym_240["smem_cd_aligned_bytes"] - asym_240["smem_cd_aligned_bytes"],
        },
        "mega_mhc_fusion": {
            "tokens": tokens,
            "hidden": hidden,
            "hc_mult": hc_mult,
            "dynamic_split_k": dynamic_splits,
            "unfused_6kernel_hbm_mib": round(unfused_hbm_bytes / (1024 * 1024), 2),
            "fused_mega_mhc_hbm_mib": round(fused_hbm_bytes / (1024 * 1024), 2),
            "hbm_traffic_reduction_pct": round((1.0 - fused_hbm_bytes / unfused_hbm_bytes) * 100.0, 2),
            "unfused_latency_us": round(unfused_mhc_us, 2),
            "fused_latency_us": round(fused_mhc_us, 2),
            "mhc_speedup_x": round(unfused_mhc_us / fused_mhc_us, 2),
        },
    }


def _check(results: dict[str, Any]) -> None:
    sweeps = results["mega_moe_tile_sweeps"]
    assert [s["block_m"] for s in sweeps] == [16, 64, 192, 240]
    assert sweeps[-1]["store_block_m_l1"] == 24
    assert sweeps[-1]["store_block_m_l2"] == 8
    comp = results["asymmetric_store_block_240_comparison"]
    assert comp["smem_cd_saved_bytes"] == 14336
    mhc = results["mega_mhc_fusion"]
    assert mhc["hbm_traffic_reduction_pct"] > 34.0
    assert mhc["mhc_speedup_x"] > 1.75


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against results.json")
    args = parser.parse_args()

    results = simulate_deepgemm_mega_moe_mhc()
    _check(results)

    out_path = Path(__file__).resolve().parent / "results.json"
    if args.check:
        expected = json.loads(out_path.read_text(encoding="utf-8"))
        assert results == expected, "Mismatch with committed results.json"
        print("Check passed: deepgemm_mega_moe_mhc_sim matches results.json")
    else:
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
