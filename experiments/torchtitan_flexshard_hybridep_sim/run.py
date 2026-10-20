#!/usr/bin/env python3
r"""Deterministic simulator for PyTorch TorchTitan FlexShard (`ComputeLayout`, `BlockShard`,
`Owned`) optimizer resharding, DistMuon Newton-Schulz communication overlap, and
GB200 NVLink72 HybridEP non-blocking dispatch capacity sizing.

Models:
1. `BlockShard` heterogeneous periodic matrix partitioning (`optimizer_reshard.py`):
   - `num_blocks(dim_size) = (dim_size // sum(block_sizes)) * len(block_sizes)`
   - `block_start(block_index) = (block_index // len(block_sizes)) * sum(block_sizes) + sum(block_sizes[:offset])`
   - Evaluates Kimi `wkv_a` `(512, 64)` heterogeneous KV-latent + RoPE-key block sharding
     vs naive uniform row sharding (which corrupts Newton-Schulz orthogonalization across matrix boundaries).
2. Bucketed storage-to-compute redistribution (`_optimizer_reshard_schedule.py`, `dist_muon.py`):
   - FP32 persistent DTensor storage -> BF16 (`compute_dtype=torch.bfloat16`) packed All-to-All / AllGather
     redistribution -> independent per-block Newton-Schulz orthogonalization with aspect-ratio LR scaling
     $\sqrt{\max(1, R / C)}$.
3. GB200 NVLink72 HybridEP non-blocking token dispatch (`torchtitan/distributed/deepep/hybridep.py`):
   - `_num_permuted_tokens_for_non_blocking`:
     `n = int(num_tokens * ep_size * min(num_local_experts, top_k) * capacity_factor)`
     aligned up to `pad_multiple` (e.g., 128 for MXFP8).
   - Evaluates memory savings, D2H synchronization elimination (`cudaStreamSynchronize`), and sticky
     `_over_budget` token-drop behavior across routing skew profiles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


class BlockShardModel:
    """Replicate `BlockShard` from `torchtitan/distributed/flex_shard/optimizer_reshard.py`."""

    def __init__(self, dim: int, block_sizes: tuple[int, ...]) -> None:
        if not block_sizes or any(s <= 0 for s in block_sizes):
            raise ValueError("BlockShard.block_sizes must be a nonempty tuple of positive integers")
        self.dim = dim
        self.block_sizes = block_sizes

    def num_blocks(self, dim_size: int) -> int:
        period = sum(self.block_sizes)
        num_periods, remainder = divmod(dim_size, period)
        if remainder:
            raise ValueError(
                f"BlockShard dimension size {dim_size} must be divisible by block-size period {period}"
            )
        return num_periods * len(self.block_sizes)

    def block_start(self, block_index: int) -> int:
        num_periods, offset = divmod(block_index, len(self.block_sizes))
        return num_periods * sum(self.block_sizes) + sum(self.block_sizes[:offset])


def simulate_kimi_wkva_block_sharding(
    num_Repetitions: int = 8,
    block_sizes: tuple[int, ...] = (512, 64),
    cols: int = 4096,
    num_ranks: int = 4,
) -> dict[str, Any]:
    """Simulate `BlockShard(0, (512, 64))` on Kimi shared `wkv_a` projection."""
    bs = BlockShardModel(0, block_sizes)
    dim_size = num_Repetitions * sum(block_sizes)
    total_blocks = bs.num_blocks(dim_size)
    blocks_per_rank = total_blocks // num_ranks

    rank_partitions: list[dict[str, Any]] = []
    for r in range(num_ranks):
        b_start = r * blocks_per_rank
        b_end = (r + 1) * blocks_per_rank if r < num_ranks - 1 else total_blocks
        row_start = bs.block_start(b_start)
        row_end = bs.block_start(b_end)
        local_blocks = []
        for b_idx in range(b_start, b_end):
            r0 = bs.block_start(b_idx)
            r1 = bs.block_start(b_idx + 1)
            rows_b = r1 - r0
            aspect_scale = math.sqrt(max(1.0, rows_b / float(cols)))
            local_blocks.append(
                {
                    "block_index": b_idx,
                    "row_range": [r0, r1],
                    "rows": rows_b,
                    "cols": cols,
                    "muon_aspect_lr_scale": round(aspect_scale, 6),
                }
            )
        rank_partitions.append(
            {
                "rank": r,
                "block_range": [b_start, b_end],
                "row_start": row_start,
                "row_end": row_end,
                "local_rows": row_end - row_start,
                "num_independent_matrices": b_end - b_start,
                "sample_blocks": local_blocks[:2],
            }
        )

    # Check whether naive uniform row sharding (e.g. across 6 ranks) splits a 512 or 64 block mid-matrix
    naive_ranks = 6
    naive_rows_per_rank = dim_size // naive_ranks
    valid_boundaries = {bs.block_start(i) for i in range(total_blocks + 1)}
    broken_boundaries = [
        r * naive_rows_per_rank
        for r in range(1, naive_ranks)
        if (r * naive_rows_per_rank) not in valid_boundaries
    ]

    return {
        "dim_size": dim_size,
        "cols": cols,
        "block_sizes": list(block_sizes),
        "total_blocks": total_blocks,
        "rank_partitions": rank_partitions,
        "naive_6rank_broken_boundaries": broken_boundaries,
        "naive_6rank_boundary_violations": len(broken_boundaries),
    }


def simulate_bucketed_muon_overlap(
    num_buckets: int = 4,
    params_per_bucket: int = 8,
    matrix_shape: tuple[int, int] = (4096, 4096),
    mesh_size: int = 8,
    interconnect_gbps: float = 900.0,
    tflops_bf16: float = 1800.0,
    ns_steps: int = 5,
) -> dict[str, Any]:
    """Compare FP32 unbucketed sequential reshard vs FlexShard BF16 bucketed overlapped schedule."""
    rows, cols = matrix_shape
    numel_per_param = rows * cols
    bucket_numel = params_per_bucket * numel_per_param

    # Storage is FP32 (4B), ComputeLayout.compute_dtype is BF16 (2B)
    fp32_gather_bytes = bucket_numel * 4 * ((mesh_size - 1) / mesh_size)
    bf16_reshard_bytes = bucket_numel * 2 * ((mesh_size - 1) / mesh_size)

    # Round-trip communication (inputs to compute owner + directions back)
    comm_ms_fp32_per_bucket = (2.0 * fp32_gather_bytes / (interconnect_gbps * 1e9)) * 1e3
    comm_ms_bf16_per_bucket = (2.0 * bf16_reshard_bytes / (interconnect_gbps * 1e9)) * 1e3

    # Newton-Schulz compute on compute-sharded owner (each rank computes 1/mesh_size of bucket matrices)
    # Each NS iteration does 3 matrix multiplications of size [R, R] / [R, C] (~6 * R^2 * C FLOPs)
    flops_per_matrix = ns_steps * 6.0 * (rows**2) * cols
    compute_flops_per_bucket_per_rank = (params_per_bucket / mesh_size) * flops_per_matrix
    compute_ms_per_bucket = (compute_flops_per_bucket_per_rank / (tflops_bf16 * 1e12)) * 1e3

    sequential_fp32_total_ms = num_buckets * (comm_ms_fp32_per_bucket + compute_ms_per_bucket)
    # Overlapped pipeline: first bucket forward comm + N buckets max(compute, comm) + last bucket return comm
    single_leg_bf16_ms = 0.5 * comm_ms_bf16_per_bucket
    overlapped_bf16_total_ms = (
        single_leg_bf16_ms
        + (num_buckets - 1) * max(compute_ms_per_bucket, comm_ms_bf16_per_bucket)
        + compute_ms_per_bucket
        + single_leg_bf16_ms
    )
    speedup = sequential_fp32_total_ms / overlapped_bf16_total_ms

    return {
        "num_buckets": num_buckets,
        "params_per_bucket": params_per_bucket,
        "mesh_size": mesh_size,
        "fp32_comm_ms_per_bucket": round(comm_ms_fp32_per_bucket, 4),
        "bf16_comm_ms_per_bucket": round(comm_ms_bf16_per_bucket, 4),
        "ns_compute_ms_per_bucket": round(compute_ms_per_bucket, 4),
        "sequential_fp32_total_ms": round(sequential_fp32_total_ms, 4),
        "flexshard_overlapped_bf16_total_ms": round(overlapped_bf16_total_ms, 4),
        "step_speedup_ratio": round(speedup, 4),
        "comm_volume_reduction_pct": 50.0,
    }


def num_permuted_tokens_for_non_blocking(
    num_tokens: int,
    ep_size: int,
    num_local_experts: int,
    top_k: int,
    moe_expert_capacity_factor: float,
    pad_multiple: int | None = None,
) -> int:
    """Exact replica of `_num_permuted_tokens_for_non_blocking` in `hybridep.py`."""
    n = int(
        num_tokens
        * ep_size
        * min(num_local_experts, top_k)
        * moe_expert_capacity_factor
    )
    if pad_multiple is not None:
        n = ((n + pad_multiple - 1) // pad_multiple) * pad_multiple
    return n


def simulate_hybridep_non_blocking(
    num_tokens: int = 4096,
    ep_size: int = 72,
    num_local_experts: int = 4,
    top_k: int = 8,
    hidden_dim: int = 7168,
    pad_multiple: int = 128,
) -> dict[str, Any]:
    """Evaluate GB200 NVLink72 HybridEP blocking vs non-blocking capacity factors."""
    # Expected tokens per rank under perfectly balanced routing:
    # Each of `ep_size` ranks sends `num_tokens * top_k` tokens across `ep_size * num_local_experts` total experts.
    # A single rank owns `num_local_experts` experts, so balanced tokens received = num_tokens * top_k.
    balanced_received_tokens = num_tokens * top_k
    skewed_p99_received_tokens = int(balanced_received_tokens * 1.35)

    capacity_sweep: list[dict[str, Any]] = []
    for cf in (0.125, 0.25, 0.35, 0.5, 1.0):
        buf_tokens = num_permuted_tokens_for_non_blocking(
            num_tokens=num_tokens,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            top_k=top_k,
            moe_expert_capacity_factor=cf,
            pad_multiple=pad_multiple,
        )
        # BF16 activations (2B) + FP32 scores (4B)
        buf_mib = (buf_tokens * (hidden_dim * 2 + 4)) / (1024.0 * 1024.0)
        drops_balanced = max(0, balanced_received_tokens - buf_tokens)
        drops_skewed = max(0, skewed_p99_received_tokens - buf_tokens)
        capacity_sweep.append(
            {
                "capacity_factor": cf,
                "allocated_permuted_tokens": buf_tokens,
                "buffer_mib": round(buf_mib, 2),
                "balanced_received_tokens": balanced_received_tokens,
                "skewed_p99_received_tokens": skewed_p99_received_tokens,
                "over_budget_balanced": drops_balanced > 0,
                "over_budget_skewed_p99": drops_skewed > 0,
                "dropped_tokens_skewed_p99": drops_skewed,
            }
        )

    return {
        "ep_size": ep_size,
        "num_local_experts": num_local_experts,
        "top_k": top_k,
        "pad_multiple": pad_multiple,
        "balanced_expected_tokens": balanced_received_tokens,
        "capacity_factor_sweep": capacity_sweep,
    }


def run_simulation() -> dict[str, Any]:
    return {
        "Metadata": {
            "experiment": "torchtitan_flexshard_hybridep_sim",
            "upstream_repo": "pytorch/torchtitan",
            "upstream_commit": "3b1714ab03e15d1dfe9fa824ec4b5739e45a40c0",
            "target_arch": "GB200 NVLink72 / PyTorch DTensor FlexShard",
        },
        "kimi_wkva_block_sharding": simulate_kimi_wkva_block_sharding(),
        "bucketed_muon_overlap": simulate_bucketed_muon_overlap(),
        "hybridep_non_blocking": simulate_hybridep_non_blocking(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate deterministic results against committed results.json",
    )
    args = parser.parse_args()

    results = run_simulation()
    formatted = json.dumps(results, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not RESULTS_PATH.exists():
            sys.stderr.write(f"Missing {RESULTS_PATH}\n")
            return 1
        existing = RESULTS_PATH.read_text(encoding="utf-8")
        if existing != formatted:
            sys.stderr.write("results.json mismatch for torchtitan_flexshard_hybridep_sim\n")
            return 1
        print("torchtitan_flexshard_hybridep_sim: --check PASSED")
        return 0

    RESULTS_PATH.write_text(formatted, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
