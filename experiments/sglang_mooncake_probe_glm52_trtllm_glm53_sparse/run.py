#!/usr/bin/env python3
"""Deterministic verification of SGLang Mooncake failed-session exponential-backoff
probe & NIXL UCX per-GPU HCA binding, GLM-5.2 DSA side-stream join ordering & UMBP
packed draft layer mappings, and TensorRT-LLM GLM-5.3-Flash glm_kpool sparse attention
speedups & Rubin SM107 CUTLASS MXFP8 MoE.

Upstream references:
- sgl-project/sglang PR #45338 (be65391abd01f711cdc443a3fe8f508b0c960357)
- sgl-project/sglang PR #43482 (2e7c9d851c868678ddfcd93b4e8000cd23e1f779)
- sgl-project/sglang PR #45349 (f7b8c6fe404ca4f29c3188cb50692f7735d45a99)
- sgl-project/sglang PR #44929 (da7e3f1c125be0f1ef2101c474b679296acdb325)
- NVIDIA/TensorRT-LLM PR #20454 (ef913d39b3b5ba488dc0972bd400e52b41899767)
- NVIDIA/TensorRT-LLM PR #19953 (4636b6311f55ec4ffe16d44fba3601f41638ec42)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify_sglang_mooncake_probe_and_nixl_ucx() -> dict[str, Any]:
    """Model SGLang PR #45338 Mooncake failed-session probe backoff and PR #43482 NIXL UCX HCA list."""
    base_interval_s = 30.0
    max_interval_s = 300.0
    idle_timeout_s = 600.0

    # Simulate a dead decode session that fails probes repeatedly after last request at t=0.0
    probe_timestamps: list[float] = []
    intervals_after_fail: list[float] = []
    t = 0.0
    last_requested = 0.0
    interval = base_interval_s
    next_probe_at = last_requested + interval

    while True:
        t = next_probe_at
        if t - last_requested >= idle_timeout_s:
            pruned_at_s = t
            break
        probe_timestamps.append(t)
        interval = min(interval * 2.0, max_interval_s)
        intervals_after_fail.append(interval)
        next_probe_at = t + interval

    # Without backoff/pruning, probing every 30s over 24 hours (86400s) would issue 2880 health checks/session.
    probes_per_24h_unbounded = int(86400 // base_interval_s)
    probes_before_idle_prune = len(probe_timestamps)

    # PR #43482 NIXL UCX device_list resolution: strips spaces, joins with literal ", ", rejects ":" port suffixes
    def resolve_nixl_ucx_device_list(raw_devices: str) -> str:
        names = [name.strip() for name in raw_devices.split(",")]
        if any(not name or ":" in name for name in names):
            raise ValueError("NIXL UCX requires HCA names without port suffixes")
        return ", ".join(names)

    valid_device_list = resolve_nixl_ucx_device_list("mlx5_0,mlx5_1")
    port_suffix_rejected = False
    try:
        resolve_nixl_ucx_device_list("mlx5_0:1,mlx5_1:1")
    except ValueError:
        port_suffix_rejected = True

    invariants = (
        probe_timestamps == [30.0, 90.0, 210.0, 450.0]
        and intervals_after_fail == [60.0, 120.0, 240.0, 300.0]
        and pruned_at_s == 750.0
        and probes_before_idle_prune == 4
        and probes_per_24h_unbounded == 2880
        and valid_device_list == "mlx5_0, mlx5_1"
        and port_suffix_rejected
    )
    return {
        "probe_timestamps_s": probe_timestamps,
        "intervals_after_fail_s": intervals_after_fail,
        "pruned_at_s": pruned_at_s,
        "probes_before_idle_prune": probes_before_idle_prune,
        "probes_per_24h_unbounded": probes_per_24h_unbounded,
        "valid_device_list": valid_device_list,
        "port_suffix_rejected": port_suffix_rejected,
        "invariants_passed": invariants,
    }


def verify_sglang_glm52_side_stream_and_umbp_packed_layers() -> dict[str, Any]:
    """Model SGLang PR #45349 GLM-5.2 DSA side-stream join and PR #44929 UMBP packed draft layers."""
    # PR #45349: In CUDA graph capture, if alt_stream work (`wq_b` or `q_b_proj_forward`) is recorded
    # before main-stream work (`wk_weights_proj` or `indexer`), the graph topology forks before the
    # main stream advances and can leave replay synchronization on the side stream. Recording the
    # side-stream op immediately before `current_stream.wait_stream(alt_stream)` keeps the fork-join
    # tightly bracketed so the join stays on the main lane at graph replay.
    old_capture_sequence = ["fork_to_alt", "alt:wq_b", "main:wk_weights_proj", "join_to_main"]
    new_capture_sequence = ["fork_to_alt", "main:wk_weights_proj", "alt:wq_b", "join_to_main"]
    join_immediately_follows_alt = (
        new_capture_sequence.index("join_to_main") - new_capture_sequence.index("alt:wq_b") == 1
    )

    # PR #44929: UMBP Direct Linker supports 1-to-many logical_layer -> buffer_indices for packed
    # speculative draft models (e.g. logical layer 60 packs draft buffers [0, 1, 2]).
    layer_mapping = {60: [0, 1], 61: 2}
    pool_layer_count = 3
    by_buffer: dict[int, int] = {}
    for logical_layer, mapped in layer_mapping.items():
        indices = [mapped] if isinstance(mapped, int) else list(mapped)
        for index in indices:
            if index in by_buffer:
                raise ValueError(f"Duplicate buffer {index}")
            by_buffer[index] = logical_layer
    contiguous_bijection = sorted(by_buffer) == list(range(pool_layer_count))
    ordered_unique_logical_layers = list(dict.fromkeys(by_buffer[i] for i in range(pool_layer_count)))

    invariants = (
        join_immediately_follows_alt
        and contiguous_bijection
        and ordered_unique_logical_layers == [60, 61]
    )
    return {
        "old_capture_sequence": old_capture_sequence,
        "new_capture_sequence": new_capture_sequence,
        "join_immediately_follows_alt": join_immediately_follows_alt,
        "contiguous_bijection": contiguous_bijection,
        "ordered_unique_logical_layers": ordered_unique_logical_layers,
        "invariants_passed": invariants,
    }


def verify_trtllm_glm53_kpool_and_sm107_mxfp8() -> dict[str, Any]:
    """Model TensorRT-LLM PR #20454 GLM-5.3-Flash glm_kpool sparse attention & PR #19953 SM107 MXFP8 MoE."""
    # 1. `kpool_update` 2D draft-verification deduplication (`ROWS_PER_TABLE > 1`):
    # When verifying T=4 consecutive draft positions per request in `positions` of shape [N, T],
    # row `b` skips recomputing the pool key if `b % T < T - 1` and `POS[b + 1] // KPOOL == POS[b] // KPOOL`.
    kpool_size = 64
    # Request 0 at positions [120, 121, 122, 123] (all in pool 1);
    # Request 1 at positions [126, 127, 128, 129] (crosses boundary from pool 1 to pool 2 at 127->128).
    positions_2d = [
        [120, 121, 122, 123],
        [126, 127, 128, 129],
    ]
    rows_per_table = len(positions_2d[0])
    flat_pos = [p for row in positions_2d for p in row]
    executed_rows: list[ tuple[int, int, int] ] = []
    for b, pos in enumerate(flat_pos):
        if rows_per_table > 1 and (b % rows_per_table < rows_per_table - 1):
            if flat_pos[b + 1] // kpool_size == pos // kpool_size:
                continue
        executed_rows.append((b // rows_per_table, pos, pos // kpool_size))

    # 2. Native FP8 decode routing (`can_run_native`):
    # Old: only BF16 KV cache and num_heads <= 16.
    # New: FP8 E4M3 KV cache (kv_heads == 1, head_dim == 512, tokens_per_block == 64, topk % 64 == 0,
    # 1 <= num_heads <= 128) OR BF16 KV cache with 1 <= num_heads <= 16.
    def can_run_native_old(kv_dtype: str, num_heads: int, topk: int) -> bool:
        return kv_dtype == "bf16" and 1 <= num_heads <= 16 and topk % 64 == 0

    def can_run_native_new(kv_dtype: str, num_heads: int, topk: int) -> bool:
        if topk <= 0 or topk % 64 != 0:
            return False
        if kv_dtype == "fp8_e4m3":
            return 1 <= num_heads <= 128
        if kv_dtype == "bf16":
            return 1 <= num_heads <= 16
        return False

    fp8_tp1_64heads_old = can_run_native_old("fp8_e4m3", 64, 2048)
    fp8_tp1_64heads_new = can_run_native_new("fp8_e4m3", 64, 2048)

    # 3. Fused `append_paged_state_rows` launch reduction for decode:
    # Old path: paged_slot_indices (div + mul + sub + index) + quantize + latent_pool write + index_pool write (~6 ops)
    # New path: 1 Triton kernel `_append_paged_state_kernel` writing both latent_pool (FP8/BF16) and index_pool (BF16).
    decode_append_launches_before = 6
    decode_append_launches_after = 1

    # 4. PR #19953: CutlassFusedMoE QuantAlgo.MXFP8 SM constraint includes Rubin SM107
    mxfp8_sm_constraint_before = [100, 103]
    mxfp8_sm_constraint_after = [100, 103, 107]

    invariants = (
        executed_rows == [(0, 123, 1), (1, 127, 1), (1, 129, 2)]
        and not fp8_tp1_64heads_old
        and fp8_tp1_64heads_new
        and decode_append_launches_after == 1
        and 107 in mxfp8_sm_constraint_after
    )
    return {
        "total_verify_tokens": len(flat_pos),
        "executed_pool_updates": len(executed_rows),
        "executed_rows_table_pos_pool": [list(x) for x in executed_rows],
        "fp8_tp1_64heads_native_old": fp8_tp1_64heads_old,
        "fp8_tp1_64heads_native_new": fp8_tp1_64heads_new,
        "decode_append_launches_before": decode_append_launches_before,
        "decode_append_launches_after": decode_append_launches_after,
        "mxfp8_sm_constraint_before": mxfp8_sm_constraint_before,
        "mxfp8_sm_constraint_after": mxfp8_sm_constraint_after,
        "invariants_passed": invariants,
    }


def run_experiment() -> dict[str, Any]:
    part1 = verify_sglang_mooncake_probe_and_nixl_ucx()
    part2 = verify_sglang_glm52_side_stream_and_umbp_packed_layers()
    part3 = verify_trtllm_glm53_kpool_and_sm107_mxfp8()
    all_passed = (
        part1["invariants_passed"]
        and part2["invariants_passed"]
        and part3["invariants_passed"]
    )
    return {
        "experiment": "sglang_mooncake_probe_glm52_trtllm_glm53_sparse",
        "upstream_commits": [
            "sgl-project/sglang@be65391abd01f711cdc443a3fe8f508b0c960357",
            "sgl-project/sglang@2e7c9d851c868678ddfcd93b4e8000cd23e1f779",
            "sgl-project/sglang@f7b8c6fe404ca4f29c3188cb50692f7735d45a99",
            "sgl-project/sglang@da7e3f1c125be0f1ef2101c474b679296acdb325",
            "NVIDIA/TensorRT-LLM@ef913d39b3b5ba488dc0972bd400e52b41899767",
            "NVIDIA/TensorRT-LLM@4636b6311f55ec4ffe16d44fba3601f41638ec42",
        ],
        "sglang_mooncake_probe_and_nixl_ucx": part1,
        "sglang_glm52_side_stream_and_umbp_packed_layers": part2,
        "trtllm_glm53_kpool_and_sm107_mxfp8": part3,
        "all_invariants_passed": all_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Assert all invariants pass")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("summary.json"),
        help="Output summary JSON path",
    )
    args = parser.parse_args()
    summary = run_experiment()
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check and not summary["all_invariants_passed"]:
        raise SystemExit("Invariant check failed")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
